# Quick Reference - Key Code Paths

## Entry Point

```
/home/user/slime/scripts/run-qwen3-4B.sh (Lines 1-150)
    ↓
Ray cluster setup (line 124)
    ↓
ray job submit (lines 135-150)
    ↓
/home/user/slime/train.py
    args = parse_args() (line 116)
    train(args) (line 117)
```

## Phase 1: Initialization

### 1.1 Argument Parsing
```
File: /home/user/slime/slime/utils/arguments.py
Function: parse_args() → returns Namespace with all config

Key args:
  - actor_num_nodes: 1
  - actor_num_gpus_per_node: 8
  - tensor_model_parallel_size: 2
  - advantage_estimator: "grpo"
  - num_rollout: 3000
  - rollout_batch_size: 32
  - n_samples_per_prompt: 8
```

### 1.2 Ray Placement Groups
```
File: /home/user/slime/slime/ray/placement_group.py
Lines 71-109: create_placement_groups(args)

Returns dict:
{
    "actor": (placement_group, [0,1,2,3,4,5,6,7]),
    "rollout": (placement_group, [0,1,2,3,4,5,6,7]),
    "critic": None
}
```

### 1.3 Rollout Manager Creation
```
File: /home/user/slime/slime/ray/rollout.py
Lines 34-70: RolloutManager.__init__()

Key initialization:
  - Load prompt data source
  - Initialize 4 SGLang engines (num_engines = 8/2 = 4)
  - Load rollout function: generate_rollout()
  - Setup reward postprocessing
```

### 1.4 Training Actor Creation
```
File: /home/user/slime/slime/ray/actor_group.py
Lines 11-148: RayTrainGroup

File: /home/user/slime/slime/backends/megatron_utils/actor.py
Lines 42-131: MegatronTrainRayActor.init()

Initialization sequence:
1. Line 49: monkey_patch_torch_dist()
2. Lines 60-65: Load config and tokenizer
3. Line 53: Initialize Megatron distributed groups
4. Lines 76-110: Setup model and optimizer
5. Lines 87-101: Setup weight backuper and reference model
```

### Key Data Structures During Init

```python
# From parse_args()
args = Namespace(
    actor_num_nodes=1,
    actor_num_gpus_per_node=8,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=1,
    data_parallel_size=4,  # 8 / (2*1)
    advantage_estimator="grpo",
    num_rollout=3000,
    rollout_batch_size=32,
    n_samples_per_prompt=8,
    global_batch_size=256,
    lr=1e-6,
    eps_clip=0.2,
    kl_loss_coef=0.0,
    rollout_temperature=0.8,
    ...
)

# Distributed groups for 8 GPUs:
TP_groups = [[0,1], [2,3], [4,5], [6,7]]      # size=2 each
DP_groups = [[0,1], [2,3], [4,5], [6,7]]      # size=2 each
PP_groups = [[0,1,2,3,4,5,6,7]]               # size=8 (all)
CP_groups = [[0,1,2,3,4,5,6,7]]               # size=8 (all)
```

---

## Phase 2: Main Training Loop

```
File: /home/user/slime/train.py
Lines 14-112: train(args)

Main loop structure (lines 62-104):
for rollout_id in range(args.start_rollout_id, args.num_rollout):
    # 2a. Evaluation (if due)
    if eval_interval and (rollout_id + 1) % eval_interval == 0:
        ray.get(rollout_manager.eval.remote(rollout_id))
    
    # 2b. Generate rollouts
    rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
    
    # 2c. Train actor
    ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
    
    # 2d. Checkpoint (if due)
    if save_interval and (rollout_id + 1) % save_interval == 0:
        actor_model.save_model(rollout_id)
    
    # 2e. Update weights
    actor_model.update_weights()
```

### 2b. Rollout Generation

```
File: /home/user/slime/slime/ray/rollout.py
Lines 89-103: RolloutManager.generate(rollout_id)

Flow:
1. Load prompts from dataset
2. Generate samples via SGLang engines
   - Each engine: 8 samples per prompt
   - Reward model computes reward per sample
3. Normalize rewards (group-wise for GRPO)
4. Convert to training format
5. Return ray.put(train_data)

Output data structure:
RolloutBatch = {
    "tokens": list[torch.Tensor],           # [T] per sample
    "response_lengths": list[int],
    "rewards": list[float],                 # normalized
    "raw_reward": list[float],              # unnormalized
    "loss_masks": list[torch.Tensor],
    "truncated": list[int],
    "sample_indices": list[int],
}
```

---

## Phase 3: Actor Training

```
File: /home/user/slime/slime/ray/actor_group.py
Lines 118-120: RayTrainGroup.async_train(rollout_id, rollout_data_ref)

File: /home/user/slime/slime/backends/megatron_utils/actor.py
Lines 261-274: MegatronTrainRayActor.train()
Lines 304-395: train_actor()

Training flow:
1. Fetch and preprocess data (lines 265-266)
2. Create data iterators (line 306)
3. Compute reference log probs (lines 313-322)
4. Compute current policy log probs (lines 325-337)
5. Compute advantages (lines 349-354)
6. Training steps (lines 362-372)
7. Update weight backup (line 382)
```

### 3.3 Data Iterator

```
File: /home/user/slime/slime/backends/megatron_utils/data.py
Lines 209-305: get_data_iterator()

With dynamic batch sizing (--use-dynamic-batch-size):
  - Calculate max microbatch tokens: 9216
  - Use seqlen_balancing to distribute samples
  - Create micro_batch_indices: list[list[int]]

Returns:
  - data_iterator: list of DataIterator (one per VP stage)
  - num_microbatches: list[int] with counts per step
```

### 3.4 Advantages Computation

```
File: /home/user/slime/slime/backends/megatron_utils/loss.py
Lines 192-361: compute_advantages_and_returns()

For GRPO (lines 239-243):
    rewards = torch.tensor(rewards, dtype=torch.float32)
    returns = get_grpo_returns(rewards, kl)
    advantages = [r for r in returns]

Formula:
  kl = log_probs_old - log_probs_new
  returns = rewards - kl_coef * kl
          = rewards (since kl_coef=0)
  
With reward normalization (group-wise):
  rewards -= rewards.mean(dim=-1, keepdim=True)
```

### 3.5 Forward-Backward Pass

```
File: /home/user/slime/slime/backends/megatron_utils/model.py
Lines 291-494: train_one_step()

1. Zero gradients (lines 322-325)

2. Forward pass (lines 333-436):
   output_tensor, loss_fn = forward_step(batch)

3. Pipeline execution (lines 439-449):
   losses_reduced = forward_backward_func(
       forward_step_func, data_iterator, model,
       num_microbatches, ...
   )

4. Loss computation via loss_function():
   policy_loss = PPO_clipped(ratio, advantages)
   entropy_loss = -entropy
   total_loss = policy_loss - entropy_coef * entropy_loss

5. Optimizer step (lines 463-469):
   if valid_step:
       optimizer.step()
       opt_param_scheduler.step(increment=global_batch_size)

6. Cleanup (lines 471-474):
   Zero gradients again
```

#### Loss Formula

```
L_policy = -min(ratio * advantage, clip(ratio, 1-eps, 1+eps) * advantage)

where:
  ratio = exp(log_prob_new - log_prob_old)
  advantage = returns (for GRPO)
  eps_clip = 0.2
  eps_clip_high = 0.28
```

---

## Phase 4: Weight Synchronization

```
File: /home/user/slime/slime/ray/actor_group.py
Lines 126-128: RayTrainGroup.update_weights()

File: /home/user/slime/slime/backends/megatron_utils/actor.py
Lines 403-435: MegatronTrainRayActor.update_weights()

Flow:
1. Get rollout engines and lock
2. Call weight_updater.update_weights()
```

### 4.2 Weight Update Implementation

```
File: /home/user/slime/slime/backends/megatron_utils/update_weight_utils.py
Lines 334-581: UpdateWeightFromTensor

update_weights() flow:
1. Increment weight version (line 420)
2. Flush SGLang caches (line 424)
3. For each parameter bucket (lines 430-433):
   a. Gather bucket parameters (lines 437-499)
      - Get local weights from TensorBackuper
      - Broadcast across PP ranks
      - Broadcast across EP ranks
      - All-gather TP shards → full tensors
   
   b. Convert to HF format (line 509)
   
   c. Send to engines (lines 531-581)
      - Serialize via MultiprocessingSerializer
      - Gather via Gloo on IPC rank 0
      - Send via Ray RPC to engine
```

### NCCL Communications

```
TP All-Gather:
  Rank 0 shard: [shard_0]
  Rank 1 shard: [shard_1]
  ↓ all_gather on TP group
  Rank 0: [shard_0 | shard_1]
  Rank 1: [shard_0 | shard_1]
  ↓ convert to HF format
  → send to SGLang engine

DP All-Reduce (for losses/grads):
  All ranks in DP group reduce:
  all_reduce(values, group=dp_group)
  → averaged across 2 ranks per DP group
```

---

## Phase 5: Evaluation & Checkpointing

### 5.1 Evaluation

```
File: /home/user/slime/slime/ray/rollout.py
Lines 105-116: RolloutManager.eval(rollout_id)

Triggered: every 20 rollouts (--eval-interval 20)

Flow:
1. Generate samples on eval dataset (AIME 2024)
2. Compute pass rate / accuracy
3. Save debug data (optional)
4. Log metrics to W&B
```

### 5.2 Checkpointing

```
File: /home/user/slime/slime/ray/actor_group.py
Lines 122-124: RayTrainGroup.save_model(step_id)

File: /home/user/slime/slime/backends/megatron_utils/actor.py
Lines 397-401: MegatronTrainRayActor.save_model(iteration)

Triggered: every 20 rollouts (--save-interval 20)

Flow:
1. Rank 0 of each TP group:
   save(iteration, model, optimizer, opt_param_scheduler)

2. Saves to: /root/Qwen3-4B_slime/iter_{N:07d}_zero/
   - pytorch_model.bin (HF format)
   - config.json
   - tokenizer files
   - optimizer states
```

---

## Key Data Structures

### Sample (from rollout)
```python
@dataclass
class Sample:
    tokens: List[int]                    # Tokenized prompt+response
    response_length: int                 # Length of response part
    reward: float                        # From reward model
    status: Status                       # TRUNCATED or SUCCESS
    truncated: int                       # 1 if truncated
    index: int                           # Original sample index
    loss_mask: Optional[List[int]]      # Mask for response tokens
    metadata: Optional[Dict]             # Extra data
```

### RolloutBatch (for training)
```python
@dataclass
class RolloutBatch(dict):
    tokens: List[torch.Tensor]           # [1, T] or flattened
    response_lengths: List[int]
    total_lengths: List[int]             # prompt + response
    rewards: List[float]                 # Normalized
    raw_reward: List[float]              # Unnormalized
    truncated: List[int]                 # 0/1 flags
    loss_masks: List[torch.Tensor]       # Response mask
    sample_indices: List[int]
    
    # Computed during training
    log_probs: List[torch.Tensor]        # Old policy
    ref_log_probs: List[torch.Tensor]   # Ref model
    values: List[torch.Tensor]           # Value predictions
    advantages: List[torch.Tensor]       # Computed advantages
    returns: List[torch.Tensor]          # Computed returns
```

---

## Common File Locations

| Task | File | Lines |
|------|------|-------|
| Argument parsing | arguments.py | 1-400+ |
| Ray setup | placement_group.py | 71-176 |
| Actor creation | actor_group.py | 11-149 |
| Actor init | actor.py | 42-131 |
| Main loop | train.py | 14-112 |
| Rollout gen | rollout.py | 34-260 |
| Train actor | actor.py | 304-395 |
| Data iter | data.py | 209-305 |
| Advantages | loss.py | 192-361 |
| Train step | model.py | 291-494 |
| Weight update | update_weight_utils.py | 334-581 |
| Megatron init | initialize.py | 52-106 |
| Model setup | model.py | 685-710 |

---

## Execution Checklist

```
[✓] 1. Ray cluster starts (8 GPUs)
[✓] 2. Parse arguments
[✓] 3. Create placement groups
[✓] 4. Initialize RolloutManager (4 SGLang engines)
[✓] 5. Initialize 8 training actors
    └─ Each sets up:
       - Torch distributed (NCCL)
       - Megatron model + optimizer
       - Weight backuper
       - Reference model
       
[✓] 6. Main training loop for 3000 rollouts:
    └─ For each rollout:
       a) Generate rollout data via SGLang (if needed)
       b) Train actor (all 8 ranks in parallel)
       c) Save checkpoint (every 20)
       d) Update weights to SGLang engines
       e) Run evaluation (every 20)

[✓] 7. Final model saved to /root/Qwen3-4B_slime/
```

---

## Tips for Tracing Execution

1. **For Ray remote calls**: Look for `.remote()` suffix
   - `actor.train.remote()` returns ObjectRef
   - `ray.get()` waits for result

2. **For distributed training**:
   - Ranks 0-7 work in parallel
   - TP groups: [0-1], [2-3], [4-5], [6-7]
   - DP groups: [0-1], [2-3], [4-5], [6-7]
   - Use `dist.barrier()` for synchronization

3. **For GPU communication**:
   - NCCL for GPU-to-GPU (all_gather, all_reduce, broadcast)
   - Gloo for CPU fallback
   - Ray for actor-to-actor communication

4. **For weight updates**:
   - Get weights from TensorBackuper (CPU cache)
   - All-gather TP shards
   - Serialize and send to engines
   - Engines load and update vLLM runtime

5. **For debugging**:
   - Set `--debug-train-only` to skip rollout
   - Set `--debug-rollout-only` to skip training
   - Use `log_perf_data()` for profiling
   - Check `.../debug_*.pt` files for saved data

