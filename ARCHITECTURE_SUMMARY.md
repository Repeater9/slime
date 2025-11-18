# SLIME Training Architecture - Quick Reference

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Ray Cluster (1 node, 8 GPUs)                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────────────────────┐   ┌─────────────────────────────┐  │
│  │ Training Actors (8 ranks)       │   │ SGLang Engines (4 instances)│  │
│  │ [GPU 0-7]                       │   │ [on GPUs 0,2,4,6]           │  │
│  │                                 │   │                             │  │
│  │ TP Groups:                      │   │ Inference instances:        │  │
│  │  [0-1], [2-3], [4-5], [6-7]    │   │  Engine 0: GPUs [0,1]       │  │
│  │                                 │   │  Engine 1: GPUs [2,3]       │  │
│  │ DP Groups:                      │   │  Engine 2: GPUs [4,5]       │  │
│  │  [0-1], [2-3], [4-5], [6-7]    │   │  Engine 3: GPUs [6,7]       │  │
│  │                                 │   │                             │  │
│  │ Megatron models with:           │   │ vLLM backend with:          │  │
│  │  - Hidden: 2560                 │   │  - same model architecture  │  │
│  │  - Layers: 36                   │   │  - KV cache management      │  │
│  │  - Attention heads: 32          │   │  - Batched generation       │  │
│  │  - Vocab: 151936                │   │  - Reward model integration │  │
│  └─────────────────────────────────┘   └─────────────────────────────┘  │
│           ▲   ▲   ▲   ▲                        ▲   ▲   ▲   ▲             │
│           │   │   │   │                        │   │   │   │             │
│           └─────────────────────────────────────────────────┘             │
│              NCCL Collective Communications                              │
│           (all-gather, all-reduce, broadcast)                            │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ RolloutManager Ray Actor (CPU-based)                            │    │
│  │  - Manages prompt dataset                                       │    │
│  │  - Coordinates SGLang engines                                   │    │
│  │  - Processes rollout data (reward normalization)                │    │
│  │  - Handles evaluation                                           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Flow in One Training Iteration

```
Rollout ID: N

Step 1: ROLLOUT GENERATION (SGLang on GPUs)
────────────────────────────────────────────
 
RolloutManager.generate(rollout_id=N)
    ↓
  Load prompts from dataset
    ↓
  Split into batches (batch_size=32)
    ↓
  Send to SGLang engines (4 engines in parallel)
    ↓
  Each engine generates 8 samples per prompt
    ↓
  Computes rewards (deepscaler reward model)
    ↓
  Returns: List[Sample] with {tokens, reward, response_length, ...}
    ↓
  Normalize rewards (group-wise, GRPO)
    ↓
  Convert to training format:
    {
      "tokens": list[Tensor],
      "rewards": list[float],
      "response_lengths": list[int],
      "loss_masks": list[Tensor],
      "raw_reward": list[float],
      "truncated": list[int],
    }
    ↓
  ray.put(data) → ObjectRef


Step 2: ACTOR TRAINING (Training GPUs)
──────────────────────────────────────

On each training rank [0-7]:

  Fetch ObjectRef data
    ↓
  Move tokens to GPU
    ↓
  Create data iterators (dynamic batch sizing)
    ↓
  Forward pass on reference model:
    logits_ref = model_ref(tokens)
    log_probs_ref = compute_log_probs(logits_ref)
    ↓
  Forward pass on current actor:
    logits = model_actor(tokens)
    log_probs = compute_log_probs(logits)
    ↓
  Compute advantages:
    advantages = rewards - kl_coef * (log_probs - log_probs_ref)
    [for GRPO: advantages = rewards since kl_coef=0]
    ↓
  Training step (Megatron pipeline):
    - Forward: logits = model(tokens)
    - Loss: pg_loss = PPO_clipped(logits, log_probs, advantages)
    - Backward: loss.backward()
    - All-reduce gradients across DP group
    - Optimizer step (Adam)
    ↓
  Sync with barrier
    ↓
  Continue next mini-batch


Step 3: WEIGHT SYNCHRONIZATION (All GPUs)
──────────────────────────────────────────

On rank 0 of each TP group [0,2,4,6]:

  Get trained weights from TensorBackuper
    ↓
  For each TP pair:
    - All-gather TP shards → full weight
    - Broadcast across PP ranks (if PP>1)
    - Broadcast across EP ranks (if EP>1)
    ↓
  Convert to Hugging Face format
    ↓
  Serialize to bytes
    ↓
  Gather on rank 0 of IPC group via Gloo
    ↓
  Send to corresponding SGLang engine
    ↓
  Engine loads weights, ready for next rollout


Step 4: EVALUATE & CHECKPOINT (Conditional)
─────────────────────────────────────────────

Every 20 rollouts:
  
  Evaluation:
    - Generate samples on eval dataset (AIME 2024)
    - Compute pass rate (accuracy)
    - Log metrics to W&B
  
  Checkpointing:
    - Save model on rank 0 per TP group
    - Save optimizer states
    - Save to /root/Qwen3-4B_slime/iter_{N:07d}_zero/
```

## Distributed Training Groups

### Tensor Parallel (TP=2)
```
GPU [0,1] → TP group 0
GPU [2,3] → TP group 1
GPU [4,5] → TP group 2
GPU [6,7] → TP group 3

Each pair splits model by embedding/linear layers across 2 GPUs
Communication: all_gather to reconstruct full weights
```

### Data Parallel (DP=4)
```
TP ranks [0,1] → DP group 0
TP ranks [2,3] → DP group 1
TP ranks [4,5] → DP group 2
TP ranks [6,7] → DP group 3

Each group all-reduces gradients and metrics
```

### Pipeline Parallel (PP=1)
```
Not used in this configuration
```

### Context Parallel (CP=1)
```
Not used in this configuration
```

## Key Hyperparameters

### Model Architecture
```
- Hidden size: 2560
- Num layers: 36
- Num attention heads: 32
- FFN hidden size: 9728 (SwiGLU)
- Vocab size: 151,936
- KV channels: 128
- Group query attention (num_query_groups=8)
```

### Training Configuration
```
- Learning rate: 1e-6
- Optimizer: Adam (beta1=0.9, beta2=0.98)
- Weight decay: 0.1
- LR schedule: constant
- Batch sizes:
  * Global batch: 256 samples
  * Rollout batch: 32 prompts
  * Samples per prompt: 8
```

### RL Configuration
```
- Advantage estimator: GRPO
- Reward normalization: group-wise (by prompt)
- PPO clipping: eps=0.2, eps_clip_high=0.28
- KL coefficient: 0.0 (no KL penalty)
- Entropy coefficient: 0.0 (no entropy bonus)
- Rollout temperature: 0.8
- Max response length: 8192 tokens
```

### Rollout Configuration
```
- Rollout engine: SGLang (vLLM backend)
- Num engines: 4
- GPUs per engine: 2
- Memory fraction: 0.7 (static)
- Prompt data: dapo-math-17k (17k samples)
- Num rollouts: 3000 iterations
- Save interval: every 20 rollouts
- Eval interval: every 20 rollouts
- Eval dataset: AIME 2024
- Samples per eval prompt: 16
```

## GPU Memory Management

### Colocated Architecture (--colocate flag)
```
Both training and inference run on same 8 GPUs

Training phase:
  - Training model on GPU memory
  - SGLang engines offloaded to CPU
  - Dynamic offloading via torch_memory_saver (LD_PRELOAD)

Rollout phase:
  - Training model offloaded to CPU (optional)
  - SGLang engines active on GPU memory
  - Data flows via Ray object store (CPU intermediary)
```

### Memory Optimization
```
- Weights backuper: snapshot actor/ref/old_actor weights on CPU
- Sequence packing: packed attention for variable length sequences
- Gradient accumulation: accumulate across microbatches
- Recompute: full recomputation with uniform granularity
- Dynamic batch sizing: adjust microbatch count by token budget
```

## Checkpoint Structure

```
/root/Qwen3-4B_slime/
├── iter_0000020_zero/          # After rollout 20
│   ├── pytorch_model.bin       # Model weights (TP consolidated)
│   ├── config.json            # Model config
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── tokenizer.model
├── iter_0000040_zero/          # After rollout 40
├── ...
└── latest_checkpointed_iteration.txt
```

## File Organization

```
/home/user/slime/
├── train.py                    # Main entry point
├── scripts/
│   ├── run-qwen3-4B.sh        # Shell launcher
│   └── models/
│       └── qwen3-4B.sh        # Model config
├── slime/
│   ├── ray/                   # Ray distributed setup
│   │   ├── placement_group.py
│   │   ├── actor_group.py
│   │   ├── train_actor.py
│   │   ├── rollout.py         # RolloutManager
│   │   └── rollout_data_source.py
│   ├── backends/
│   │   ├── megatron_utils/    # Megatron training backend
│   │   │   ├── actor.py       # MegatronTrainRayActor
│   │   │   ├── model.py       # Model setup & training
│   │   │   ├── loss.py        # Loss computation
│   │   │   ├── data.py        # Data iteration
│   │   │   ├── update_weight_utils.py
│   │   │   ├── initialize.py
│   │   │   └── checkpoint.py
│   │   └── sglang_utils/      # SGLang integration
│   │       └── sglang_engine.py
│   └── utils/
│       ├── arguments.py        # Argument parsing
│       ├── ppo_utils.py        # GRPO/PPO utilities
│       ├── tensor_backper.py   # Weight backup
│       └── types.py            # Data structures
```

## Key Algorithmic Components

### GRPO Loss
```
L_total = L_policy + λ_entropy * L_entropy

L_policy = -E[(log_π(a|s) - log_π_old(a|s)) * A]
         [clipped with eps=0.2]

where:
  A = (reward - kl_penalty)  for GRPO
    = reward (since kl_coef=0)
```

### Advantage Computation
```
GRPO:
  returns = reward - kl_coef * kl_divergence
  advantages = returns
  
  [with group normalization: returns -= returns.mean()]
```

### Weight Update Protocol
```
1. Gather TP shards across TP ranks via NCCL all_gather
2. Broadcast across PP/EP ranks if needed
3. Convert megatron format → HF format
4. Serialize to bytes via MultiprocessingSerializer
5. Gather across IPC group (2 ranks per engine) via Gloo
6. Send to SGLang engine via Ray RPC
7. Engine deserializes and loads into vLLM runtime
```

## Training Loop Pseudocode

```python
for rollout_id in range(0, 3000):
    # 1. Generate rollouts
    rollout_data = rollout_manager.generate(rollout_id)
    
    # 2. Train actor (on all 8 GPU ranks)
    actor_model.async_train(rollout_id, rollout_data)
    
    # 3. Save checkpoint (every 20 iterations)
    if (rollout_id + 1) % 20 == 0:
        actor_model.save_model(rollout_id)
    
    # 4. Synchronize weights to SGLang engines
    actor_model.update_weights()
    
    # 5. Run evaluation (every 20 iterations)
    if (rollout_id + 1) % 20 == 0:
        metrics = rollout_manager.eval(rollout_id)
        wandb.log(metrics)
```

---

## Quick Navigation Guide

**For understanding initialization**: See Section 1 of TRAINING_FLOW_DOCUMENTATION.md
- Ray setup: lines 71-109 in placement_group.py
- Actor init: lines 42-131 in actor.py
- Megatron setup: init() in initialize.py

**For understanding rollout generation**: See Section 2 + 6
- RolloutManager: lines 34-260 in rollout.py
- SGLang engines: sglang_utils/sglang_engine.py

**For understanding training**: See Section 3
- train_actor(): lines 304-395 in actor.py
- train_one_step(): lines 291-494 in model.py
- Loss computation: loss.py lines 192-500

**For understanding weight sync**: See Section 4
- update_weights(): lines 403-435 in actor.py
- UpdateWeightFromTensor: lines 334-581 in update_weight_utils.py

**For understanding evaluation**: See Section 5
- rollout_manager.eval(): lines 105-116 in rollout.py

**For GPU communication analysis**: See Sections 4.4 and 7
- NCCL patterns, IPC data flow
