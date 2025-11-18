# Qwen3-4B Training Process - Complete End-to-End Flow

## Overview

This document traces the complete training process when running `scripts/run-qwen3-4B.sh`, which launches a distributed RLHF training job with:
- 8 GPUs on 1 node (Ray worker)
- Tensor parallel size: 2
- Advantage estimator: GRPO
- Rollout engine: SGLang
- Training backend: Megatron-LM

---

# 1. INITIALIZATION PHASE

## 1.1 Script Entry Point and Ray Cluster Setup

**File**: `/home/user/slime/scripts/run-qwen3-4B.sh`

```bash
# Lines 4-11: Clean up any previous processes
pkill -9 sglang
ray stop --force
...

# Line 124: Start Ray cluster head node on 127.0.0.1 with 8 GPUs
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Lines 135-150: Submit training job
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 train.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 8 \
    --colocate \
    [all argument groups]
```

## 1.2 Command-line Arguments Parsing

**Function**: `parse_args()`
**File**: `/home/user/slime/slime/utils/arguments.py` (Lines 1-200+)

Parses arguments in this order:
1. **Cluster arguments** (Lines 33-109):
   - `--actor-num-nodes 1`, `--actor-num-gpus-per-node 8`
   - `--colocate` (enables offloading, training and inference on same GPUs)
   - `--num-gpus-per-node 8` (for rollout)

2. **Training backend** (Lines 113-146):
   - `--train-backend megatron` (default)
   - `--enable-weights-backuper` (save host memory)

3. **Rollout arguments** (Lines 149+):
   - `--hf-checkpoint /root/Qwen3-4B`
   - `--ref-load /root/Qwen3-4B_torch_dist` (reference model)
   - `--rollout-function-path slime.rollout.sglang_rollout.generate_rollout`
   - `--num-rollout 3000`
   - `--rollout-batch-size 32`
   - `--n-samples-per-prompt 8`
   - Reward model: `--rm-type deepscaler`

4. **GRPO-specific arguments** (from script):
   - `--advantage-estimator grpo`
   - `--eps-clip 0.2`, `--eps-clip-high 0.28`
   - `--use-kl-loss`, `--kl-loss-coef 0.00`

5. **SGLang arguments**:
   - `--rollout-num-gpus-per-engine 2`
   - `--sglang-mem-fraction-static 0.7`

6. **Megatron-LM arguments** (from script):
   - `--tensor-model-parallel-size 2`
   - `--sequence-parallel`
   - `--recompute-granularity full`
   - `--use-dynamic-batch-size`
   - `--max-tokens-per-gpu 9216`

## 1.3 Ray Placement Groups Creation

**Function**: `create_placement_groups(args)`
**File**: `/home/user/slime/slime/ray/placement_group.py` (Lines 71-109)

Creates Ray placement groups to allocate GPUs to training and rollout actors.

```
Steps:
1. Calculate total GPUs needed:
   - Since --colocate is set: num_gpus = 8 (actor GPUs only)
   - Rollout engines will use the same GPUs (colocated)
   
2. Create placement group (_create_placement_group):
   - Create placement_group with 8 bundles, each with {"GPU": 1, "CPU": 1}
   - Each bundle represents one GPU
   - Strategy: "PACK" (pack on same physical nodes)
   
3. Detect GPU IDs:
   - Use InfoActor to query actual GPU IDs from each bundle
   - Sort by node IP and GPU ID for deterministic ordering
   - Bundle indices: [0,1,2,3,4,5,6,7] on single node
   
4. Return placement group dictionary:
   {
       "actor": (pg, actor_pg_reordered_bundle_indices),    # GPUs 0-7
       "rollout": (pg, rollout_pg_reordered_bundle_indices), # GPUs 0-7
       "critic": None  # Not used in this config
   }
```

## 1.4 Rollout Manager Initialization

**Function**: `create_rollout_manager(args, pgs["rollout"])`
**File**: `/home/user/slime/slime/ray/placement_group.py` (Lines 160-176)

Creates RolloutManager as a Ray remote actor.

```
RolloutManager init flow (in slime/ray/rollout.py, Lines 34-70):

1. Initialize data source:
   - RolloutDataSourceWithBuffer reads prompt data from:
     --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   
2. Load rollout function:
   - load_function("slime.rollout.sglang_rollout.generate_rollout")
   - This function uses SGLang engines for inference
   
3. Initialize SGLang engines:
   - init_rollout_engines(args, pg, all_rollout_engines)
   - Creates num_engines = rollout_num_gpus / rollout_num_gpus_per_engine
   - Since colocate=True and rollout_num_gpus=8, rollout_num_gpus_per_engine=2:
     num_engines = 8 / 2 = 4 engines
   
4. For each engine (Lines 276-320 in rollout.py):
   - Create SGLangEngine Ray actor with:
     - num_cpus=0.2, num_gpus=0.2
     - Scheduled on bundle indices [0,2,4,6] (every 2nd GPU)
   - Engine loads HF model from /root/Qwen3-4B
   - Initializes SGLang vLLM backend
```

**Data flow**: 
- prompt_data_loader → RolloutDataSourceWithBuffer → SGLang engines (inference)

## 1.5 Training Models (Actor) Creation

**Function**: `create_training_models(args, pgs, rollout_manager)`
**File**: `/home/user/slime/slime/ray/placement_group.py` (Lines 123-157)

### Step 1: Allocate GPU resources for actor

**Class**: `RayTrainGroup` 
**File**: `/home/user/slime/slime/ray/actor_group.py` (Lines 11-149)

```
In __init__ (Lines 31-48):
- world_size = 1 * 8 = 8 (actors)
- num_gpus_per_actor = 0.4 (allows multiple actors per GPU)
- Backend selection: Megatron backend (Lines 82-91)
  - Import MegatronTrainRayActor from slime/backends/megatron_utils/__init__.py
  - MegatronTrainRayActor = ray.remote(num_gpus=1, runtime_env={...})(actor_impl)

In _allocate_gpus_for_actor (Lines 50-109):
- Create 8 TrainRayActor instances for ranks 0-7
- For each rank:
  * Schedule on placement group bundle at index [rank % 8]
  * First rank (0) gets master_addr and master_port
  * Others connect to master via MASTER_ADDR/MASTER_PORT env vars
  
Environment setup:
  - NCCL_CUMEM_ENABLE=0 (compatibility with SGLang)
  - NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1
  - LD_PRELOAD for torch_memory_saver (if offload_train_mode=tms)
```

### Step 2: Async initialization of actor

**Function**: `async_init(args, role="actor", with_ref=True)`
**File**: `/home/user/slime/slime/ray/actor_group.py` (Lines 111-116)

Returns list of async remote calls:
```python
[actor.init.remote(args, role="actor", wandb_run_id, with_ref=True) for actor in self._actor_handlers]
```

### Step 3: Detailed initialization in MegatronTrainRayActor

**Function**: `MegatronTrainRayActor.init()`
**File**: `/home/user/slime/slime/backends/megatron_utils/actor.py` (Lines 42-131)

**Phase 1: Torch distributed setup** (Lines 49-68)
```
1. monkey_patch_torch_dist() (Lines 49)
   - Patches torch.distributed for special NCCL handling
   
2. Call parent init: TrainRayActor.init() (Lines 51)
   - Set MASTER_ADDR, MASTER_PORT env vars
   - Set RANK, WORLD_SIZE env vars
   - Set LOCAL_RANK based on actual GPU IDs
   - Call torch.cuda.set_device() for current GPU
   - dist.init_process_group() with NCCL backend (Lines 60-63)
   - init_gloo_group() for CPU fallback group
   
3. NUMA affinity setup (Lines 74-82)
   - Use pynvml to bind processes to correct NUMA nodes
```

**Phase 2: Load tokenizer and config** (Lines 61-65)
```
- Serialized read to prevent concurrent write bugs
- AutoConfig.from_pretrained(args.hf_checkpoint)
  → Loads /root/Qwen3-4B/config.json
  → Stores in self.hf_config
  
- AutoTokenizer.from_pretrained(args.hf_checkpoint)
  → Loads tokenizer from Hugging Face
  → Stores in self.tokenizer
```

**Phase 3: Initialize Megatron** (Lines 53)
```
Call init(args) from slime/backends/megatron_utils/initialize.py (Lines 52-97):
  1. set_args(args) - store in megatron.training.global_vars
  2. _initialize_distributed(args) - setup tensor/pipeline/data parallel groups:
     - Tensor parallel: size 2
     - Pipeline parallel: size 1
     - Data parallel: size 8 / (2*1) = 4 groups of 2 GPUs each
     - Context parallel: size 1
     - Expert parallel: size 1
  3. _set_random_seed(args.seed) - deterministic RNG
  4. _build_tokenizer(args) - from Megatron
  5. init_num_microbatches_calculator() - for Megatron validation
```

**Phase 4: Model and optimizer initialization** (Lines 76-111)
```
Call initialize_model_and_optimizer(args, role="actor")
(slime/backends/megatron_utils/model.py, Lines 685-710):

1. setup_model_and_optimizer(args, role="actor"):
   
   a) Build model with GPTModel from Megatron:
      - Use model_provider_func from slime/backends/megatron_utils/model_provider.py
      - Creates GPT model with:
        * hidden_size=2560
        * num_layers=36
        * num_attention_heads=32
        * ffn_hidden_size=9728
        * vocab_size=151936
      - Apply tensor parallel wrapping (TP size=2)
      - Wrap with DistributedDataParallel (DDP)
   
   b) Create Megatron optimizer:
      - Type: Adam
      - LR: 1e-6
      - Beta1: 0.9, Beta2: 0.98
      - Weight decay: 0.1
      - Distributed optimizer (for large models)
   
   c) Create OptimizerParamScheduler:
      - Warmup iters based on args.lr_warmup_iters
      - Decay style: constant (--lr-decay-style constant)
      - Learn rate schedule management

2. load_checkpoint():
   - Load from --load /root/Qwen3-4B_slime/
   - Merges sharded weights across TP ranks
   - Loads optimizer states
   - Returns loaded_rollout_id (iteration number)
```

**Phase 5: Weights backup and reference model** (Lines 87-111)
```
1. Create TensorBackuper for weight snapshots:
   - self.weights_backuper = TensorBackuper.create(...)
   - Backs up "actor" weights to CPU memory
   
2. Load reference model (with_ref=True):
   - Load from --ref-load /root/Qwen3-4B_torch_dist
   - Used for KL divergence calculation
   - self.weights_backuper.backup("ref")

3. Initialize weight updater (UpdateWeightFromTensor):
   - Prepares to send weights to SGLang engines later
   - Creates IPC groups for colocated engines
   - Groups size: rollout_num_gpus_per_engine = 2
     Number of groups: 8 / 2 = 4
```

**Phase 6: Final setup** (Lines 113-129)
```
- clear_memory() - GPU cleanup
- If offload_train: sleep() - move model to CPU
- Return start_rollout_id = loaded_rollout_id + 1
```

---

# 2. MAIN TRAINING LOOP

**Function**: `train(args)` 
**File**: `/home/user/slime/train.py` (Lines 14-112)

## 2.1 Pre-training Setup

```python
# Line 27-31: Offload rollout weights if needed
if args.offload_rollout:
    ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]))

# Line 32: First weight sync to load trained weights into SGLang
actor_model.update_weights()

# Line 36-39: Load KV cache and CUDA graphs if offloading
if args.offload_rollout:
    if GPU_MEMORY_TYPE_CUDA_GRAPH is not None:
        ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH]))
    ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_KV_CACHE]))
```

## 2.2 Evaluation-only Mode (Optional)

```python
# Lines 42-43: If num_rollout=0 and eval_interval set, only do eval
if args.num_rollout == 0 and args.eval_interval is not None:
    ray.get(rollout_manager.eval.remote(rollout_id=0))
```

## 2.3 Main Loop Structure

```python
for rollout_id in range(args.start_rollout_id, args.num_rollout):  # 0-2999
```

### Iteration `rollout_id`:

#### Step 2a: Evaluation (if due)

```python
# Lines 64-65: Initial evaluation at rollout_id == 0
if args.eval_interval is not None and rollout_id == 0:
    ray.get(rollout_manager.eval.remote(rollout_id))

# Lines 106-110: Periodic evaluation
if args.eval_interval is not None and (rollout_id + 1) % args.eval_interval == 0:
    ray.get(rollout_manager.eval.remote(rollout_id))
```

#### Step 2b: Rollout Generation via SGLang

```python
# Line 67: Generate rollout data using SGLang engines
rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
```

**Detailed flow in RolloutManager.generate()** (slime/ray/rollout.py, Lines 89-103):

```
1. Call _get_rollout_data(rollout_id):
   
   a) Call rollout_generate_rollout function:
      - rollout_function_path = "slime.rollout.sglang_rollout.generate_rollout"
      - This function:
        * Sends prompts to SGLang engines
        * Each engine runs batches with:
          - num_samples_per_prompt: 8
          - rollout_temperature: 0.8
          - rollout_max_response_len: 8192
        * Generates multiple completions per prompt
        * Computes rewards using reward model (rm-type: deepscaler)
        * Returns: Sample objects with:
          - tokens (prompt + response)
          - response_length
          - reward (from reward model)
          - truncated flag
   
   b) Process returned samples:
      - Flatten list of lists to single list
      - Trim to multiple of global_batch_size (256)
      - Returns data, metrics

2. Call _post_process_rewards():
   
   - Normalize rewards:
     * Group by n_samples_per_prompt (8)
     * Subtract group mean (group normalization)
     * If grpo_std_normalization: divide by group std
   
   - Returns: raw_rewards, normalized_rewards

3. Call _convert_samples_to_train_data():
   
   - Extract training data from samples:
     * tokens: prompt + response token IDs
     * response_lengths: length of response part
     * rewards: normalized reward values
     * loss_masks: mask for response tokens
     * truncated: 1 if response was truncated
     * sample_indices: original indices for ordering
   
   - Returns dict with keys:
     {
         "tokens": list[torch.Tensor],           # prompt+response
         "response_lengths": list[int],          # response length
         "rewards": list[float],                 # normalized rewards
         "raw_reward": list[float],              # unnormalized rewards
         "truncated": list[int],                 # 1 if truncated
         "sample_indices": list[int],
         "loss_masks": list[torch.Tensor],       # response token mask
     }

4. Return ray.put(data) as Box object
   - Data stored in Ray object store for efficient access
   - Multiple actors can access without serialization overhead
```

#### Step 2c: Actor Training

```python
# Line 75: Train actor with rollout data
ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
```

This triggers parallel training across 8 GPU ranks.

#### Step 2d: Checkpointing

```python
# Lines 80-89: Save model if interval reached
if args.save_interval is not None and (rollout_id + 1) % args.save_interval == 0:
    if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps):
        actor_model.save_model(rollout_id)
    if args.use_critic:
        critic_model.save_model(rollout_id)
    if args.rollout_global_dataset:
        ray.get(rollout_manager.save.remote(rollout_id))
```

#### Step 2e: Memory Management and Weight Update

```python
# Lines 91-104: Memory and weight management
if args.enable_weights_backuper:
    offload_train()          # Move training model to CPU
    onload_rollout()         # Load rollout inference weights
    actor_model.update_weights()  # Sync actor → SGLang engines
else:
    actor_model.clear_memory()
    onload_rollout()
    actor_model.update_weights()
    offload_train()

# Re-load rollout memory for next iteration
if args.offload_rollout:
    ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH]))
    ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_KV_CACHE]))
```

---

# 3. ACTOR TRAINING PHASE

## 3.1 Async Training Entry Point

**Function**: `RayTrainGroup.async_train(rollout_id, rollout_data_ref)`
**File**: `/home/user/slime/slime/ray/actor_group.py` (Lines 118-120)

```python
return [actor.train.remote(rollout_id, rollout_data_ref) for actor in self._actor_handlers]
```

Returns list of 8 async remote calls (one per GPU rank).

## 3.2 Individual Actor Training

**Function**: `MegatronTrainRayActor.train(rollout_id, rollout_data_ref)`
**File**: `/home/user/slime/slime/backends/megatron_utils/actor.py` (Lines 261-274)

```python
def train(self, rollout_id: int, rollout_data_ref: Box) -> None:
    if self.args.offload_train:
        self.wake_up()  # Resume from CPU
    
    with timer("data_preprocess"):
        rollout_data = self._get_rollout_data(rollout_data_ref)
    
    if self.role == "critic":
        return self.train_critic(rollout_id, rollout_data)
    else:
        return self.train_actor(rollout_id, rollout_data)
```

### Data Preprocessing

**Function**: `_get_rollout_data(rollout_data_ref)`
**File**: `/home/user/slime/slime/backends/megatron_utils/actor.py` (Lines 154-186)

```
1. Fetch ray object:
   - rollout_data_ref contains data from ray.put()
   - process_rollout_data handles distributed access via DP ranks
   - Data partitioned across DP ranks (local_dp_rank, dp_size)

2. Convert tokens to GPU tensors:
   - batch["tokens"] = [torch.tensor(..., dtype=torch.long, device="cuda") for ...]
   - batch["loss_masks"] = [torch.tensor(..., dtype=torch.int, device="cuda") for ...]
   
3. Handle optional fields:
   - batch["rollout_log_probs"] (for off-policy correction)
   - batch["rollout_routed_experts"] (for routing replay)
```

### Training Actor

**Function**: `train_actor(rollout_id, rollout_data)`
**File**: `/home/user/slime/slime/backends/megatron_utils/actor.py` (Lines 304-395)

```
Main training steps:

1. Create data iterators and microbatch schedule:
   data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)
   
   (Details in section 3.3)

2. Compute log probabilities (if using advantages/returns):
   
   a) If using reference model (--ref-load):
      rollout_data.update(self.compute_log_prob(
          "ref", data_iterator, num_microbatches, store_prefix="ref_"
      ))
      → Runs forward pass on reference model
      → Computes log probs from reference policy
   
   b) Compute log probs for current actor or old_actor:
      rollout_data.update(self.compute_log_prob(
          "old_actor" if self.args.keep_old_actor else "actor",
          data_iterator, num_microbatches, store_prefix=""
      ))
      → Runs forward pass on current policy
      → Computes log probs from current policy

3. Synchronize with critic (if using critic):
   sync_actor_critic_data(self.args, rollout_data, self._actor_critic_groups)

4. Compute advantages and returns:
   compute_advantages_and_returns(self.args, rollout_data)
   
   (Details in section 3.4)

5. Training step:
   train(rollout_id, self.model, self.optimizer, 
         self.opt_param_scheduler, data_iterator, num_microbatches)
   
   (Details in section 3.5)

6. Update backup weights:
   self.weights_backuper.backup("actor")

7. Log performance data:
   log_perf_data(rollout_id, self.args)
```

## 3.3 Data Iterator Setup

**Function**: `get_data_iterator(args, model, rollout_data)`
**File**: `/home/user/slime/slime/backends/megatron_utils/data.py` (Lines 209-305)

```
Creates iterators for microbatch processing:

1. Calculate batch schedule:
   - num_local_samples = len(rollout_data["total_lengths"])
   - num_local_gbs = args.global_batch_size (256) // dp_size (4) = 64 per rank
   - num_steps_per_rollout = num_local_samples / num_local_gbs

2. If dynamic batch size (--use-dynamic-batch-size):
   
   a) Calculate max_tokens_per_microbatch:
      - args.max_tokens_per_gpu = 9216
      - Accounts for prompt + response lengths per sample
   
   b) Use seqlen_balancing to distribute samples across microbatches:
      - Ensures roughly equal token count per microbatch
      - Minimizes padding waste
      - Creates micro_batch_indices: list[list[int]]
   
   c) Create DataIterator with indices:
      data_iterator = DataIterator(rollout_data, micro_batch_indices=micro_batch_indices)

3. Return:
   - data_iterator: list of DataIterator (one per VP stage)
   - num_microbatches: list[int] with microbatch counts per step
```

**Microbatch retrieval in get_batch()** (Lines 22-84):

```
DataIterator.get_next(["tokens", "total_lengths", "response_lengths"]):

1. Fetch requested keys from rollout_data for current microbatch
2. Get batch dict with token lists
3. Apply context parallelism (CP) slicing if cp_size > 1
4. Concatenate tokens and pad to multiple of 128
5. Create PackedSeqParams for packed sequence processing
   - cu_seqlens: cumulative sequence lengths for packed format
   - max_seqlen: maximum sequence length in batch
   - qkv_format: "thd" (token, head, dimension)
```

## 3.4 Advantages and Returns Computation

**Function**: `compute_advantages_and_returns(args, rollout_data)`
**File**: `/home/user/slime/slime/backends/megatron_utils/loss.py` (Lines 192-361)

### GRPO Advantage Estimation

For `--advantage-estimator grpo`:

```python
rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
returns = get_grpo_returns(rewards, kl)
advantages = [r for r in returns]
```

**get_grpo_returns()** (slime/utils/ppo_utils.py):

```
GRPO return = reward - kl_coef * KL_divergence

Where:
  - reward: per-token reward from reward model (already normalized)
  - KL: approx_kl(log_prob_old, log_prob_new)
    = log_prob_old - log_prob_new
  - kl_coef: weight for KL term (--kl-loss-coef 0.00, so no KL penalty)
```

### Reward Normalization

In RolloutManager._post_process_rewards() (slime/ray/rollout.py, Lines 179-204):

```
Group-wise normalization (for GRPO):
  1. Reshape rewards: [total_samples] → [num_prompts, n_samples_per_prompt]
  2. Subtract group mean: rewards -= rewards.mean(dim=-1, keepdim=True)
  3. If grpo_std_normalization: divide by group std
     rewards /= (rewards.std(dim=-1, keepdim=True) + 1e-6)
  4. Flatten back to [total_samples]

This normalizes rewards within each prompt's samples.
```

### Advantage Normalization

If `--normalize-advantages`:

```python
if args.normalize_advantages:
    all_advs = torch.cat(advantages)
    all_masks = torch.cat(loss_masks)
    
    # Whitened across data-parallel group
    whitened_advs = distributed_masked_whiten(
        all_advs, all_masks,
        process_group=dp_group,
        shift_mean=True
    )
    
    # Split back to per-sample format
    chunk_lengths = [chunk.size(0) for chunk in advantages]
    advantages = list(torch.split(whitened_advs_flat, chunk_lengths))
```

## 3.5 Forward-Backward Training Step

**Function**: `train_one_step()`
**File**: `/home/user/slime/slime/backends/megatron_utils/model.py` (Lines 291-494)

### Phase 1: Setup (Lines 322-335)

```python
# Zero gradients
for model_chunk in model:
    model_chunk.zero_grad_buffer()
optimizer.zero_grad()
```

### Phase 2: Forward Pass (Lines 333-436)

Define forward_step function that:

```python
def forward_step(data_iterator, model, return_schedule_plan=False):
    batch = get_batch(data_iterator, [
        "tokens", "packed_seq_params", "total_lengths",
        "response_lengths", "loss_masks",
        "log_probs", "ref_log_probs", "values",
        "advantages", "returns", "rollout_log_probs"
    ])
    
    output_tensor = model(
        input_ids=batch["tokens"],      # [1, T]
        position_ids=None,
        attention_mask=None,
        labels=None,
        packed_seq_params=batch["packed_seq_params"]
    )
    
    return output_tensor, partial(loss_function, args, batch, num_microbatches)
```

### Phase 3: Pipeline Forward-Backward (Lines 439-449)

```python
forward_backward_func = get_forward_backward_func()

losses_reduced = forward_backward_func(
    forward_step_func=forward_step,
    data_iterator=data_iterator,
    model=model,
    num_microbatches=num_microbatches,
    seq_length=args.seq_length,
    micro_batch_size=args.micro_batch_size,
    forward_only=False,
    # collect_non_loss_data=False
)
```

**Pipeline execution**:
- Megatron's GPT-forward-backward splits computation into microbatches
- Runs forward and backward passes with gradient accumulation
- Returns list of loss dicts per microbatch

### Phase 4: Loss Computation

**Function**: `loss_function()`
**File**: `/home/user/slime/slime/backends/megatron_utils/loss.py` (Lines 500+)

For GRPO loss (policy loss):

```python
def policy_loss_function(args, batch, logits, sum_of_sample_mean):
    # 1. Compute current log probs and entropy
    log_probs_and_entropy = get_log_probs_and_entropy(
        logits, args=args, unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        with_entropy=True
    )
    
    # 2. Get log probs with temperature scaling
    logits = logits / args.rollout_temperature  # temperature 0.8
    log_probs = calculate_log_probs_and_entropy(logits, tokens, tp_group)
    
    # 3. Compute PPO-style loss with clipping
    pg_loss, pg_clipfrac = compute_policy_loss(
        ppo_kl,        # old_log_probs - log_probs
        advantages,
        eps_clip=0.2,
        eps_clip_high=0.28
    )
    
    # 4. Add entropy regularization (coef=0.00)
    entropy_loss = torch.cat(entropy, dim=0)
    loss = pg_loss - args.entropy_coef * entropy_loss
    
    # 5. KL loss (use_kl_loss=True, but kl_loss_coef=0.00)
    if args.use_kl_loss:
        kl_loss = ...  # computed but multiplied by 0
    
    return loss, metrics
```

**Policy loss (PPO clipped)**:

```
L_policy = -min(ratio * advantage, clip(ratio, 1-eps, 1+eps) * advantage)

Where:
  - ratio = exp(log_prob_new - log_prob_old)
  - advantage: computed from GRPO returns
  - eps_clip = 0.2, eps_clip_high = 0.28
```

### Phase 5: Optimizer Step (Lines 451-469)

```python
# Check for NaN/Inf
found_inf_flag = optimizer.prepare_grads()
if not found_inf_flag:
    grad_norm = optimizer.get_grad_norm()
    valid_step = not torch.isnan(grad_norm) or torch.isinf(grad_norm)

if valid_step:
    # Update model parameters
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
    
    # Update learning rate (with constant schedule)
    opt_param_scheduler.step(increment=args.global_batch_size)
```

**Distributed synchronization**:
```python
# All-reduce gradients across DP group (4 ranks)
torch.distributed.all_reduce(values, group=mpu.get_data_parallel_group(...))

# Average loss values
loss_reduced[key] = value * cp_size / num_samples_or_tokens
```

### Phase 6: Cleanup (Lines 471-494)

```python
# Zero out gradient buffers
for model_chunk in model:
    model_chunk.zero_grad_buffer()
optimizer.zero_grad()

# Return reduced losses (only on last PP stage)
return loss_reduced, grad_norm
```

---

# 4. WEIGHT SYNCHRONIZATION PHASE

## 4.1 Update Weights Flow

**Function**: `actor_model.update_weights()`
**File**: `/home/user/slime/slime/ray/actor_group.py` (Lines 126-128)

```python
return ray.get([actor.update_weights.remote() for actor in self._actor_handlers])
```

Synchronously calls `update_weights()` on all 8 actor ranks.

## 4.2 Individual Actor Weight Update

**Function**: `MegatronTrainRayActor.update_weights()`
**File**: `/home/user/slime/slime/backends/megatron_utils/actor.py` (Lines 403-435)

### Phase 1: Fetch Rollout Engines

```python
rollout_engines, rollout_engine_lock, num_new_engines = ray.get(
    self.rollout_manager.get_rollout_engines_and_lock.remote()
)

# num_new_engines > 0 if engines restarted after failure
if num_new_engines > 0:
    self.weight_updater.connect_rollout_engines(rollout_engines, rollout_engine_lock)
    dist.barrier(group=get_gloo_group())
```

### Phase 2: Gather and Convert Weights

**Class**: `UpdateWeightFromTensor` (colocated scenario)
**File**: `/home/user/slime/slime/backends/megatron_utils/update_weight_utils.py` (Lines 334-581)

```
self.weight_updater.update_weights():

1. Increment weight version:
   self.weight_version += 1

2. Flush SGLang caches (rank 0):
   ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])

3. For each parameter bucket (Lines 430-433):
   a) Get locally stored weights:
      weights = self.weights_getter()  # From TensorBackuper
      
   b) Gather bucket parameters (Lines 437-499):
      - Broadcast params across PP ranks (if PP > 1)
      - Broadcast expert params across EP ranks
      - Async all_gather TP-sharded params to full tensors
        * For expert params: use expert-TP group
        * For others: use regular-TP group
      - Handles GLU rechunking for linear_fc1
      
   c) Convert to Hugging Face format:
      For each gathered param:
        - remove_padding() for embedding/output layers
        - convert_to_hf() to HF parameter format
        - Returns list of (name, tensor) tuples
   
   d) Send to engines (Lines 531-581):
      
      For colocated engines:
        - Gather weights on rank 0 of each IPC group:
          * Create groups of rollout_num_gpus_per_engine (2) ranks
          * Each group corresponds to one SGLang engine
          * Group 0: ranks [0,1] → engine 0
          * Group 1: ranks [2,3] → engine 1
          * Group 2: ranks [4,5] → engine 2
          * Group 3: ranks [6,7] → engine 3
        
        - Serialize weights:
          * Convert to MultiprocessingSerializer format (string)
          * Fits large tensors into single message
        
        - Gather via Gloo (CPU fallback):
          dist.gather_object(serialized_tensors, ...)
        
        - Send to engine via Ray:
          self._ipc_engine.update_weights_from_tensor.remote(
              serialized_named_tensors=...,
              weight_version=str(self.weight_version)
          )

4. Synchronize across all ranks:
   dist.barrier(group=get_gloo_group())
```

## 4.3 SGLang Engine Weight Update

**In SGLang Engine actor** (slime/backends/sglang_utils/sglang_engine.py):

```python
@ray.remote
class SGLangEngine:
    def update_weights_from_tensor(self, serialized_named_tensors, weight_version):
        # 1. Deserialize weights
        # 2. Load into vLLM backend
        # 3. Update weight_version
        # 4. Ready for next rollout
```

## 4.4 GPU Communication Analysis

### Tensor Parallel All-Gather

For each TP-sharded parameter (TP size = 2):

```
Rank 0 (TP rank 0):     [shard_0]
Rank 1 (TP rank 1):     [shard_1]
         ↓ (NCCL all_gather)
Rank 0:  [shard_0 | shard_1] (concatenated)
Rank 1:  [shard_0 | shard_1] (concatenated)
         ↓ (convert to HF format)
Rank 0,1: feed to SGLang engine
```

### IPC Data Flow

```
Training GPU 0,1:  [weights for engine 0]  → Gloo gather on rank 0
                        ↓
                   Serialize to string
                        ↓
                   Ray IPC to SGLang engine 0
                        ↓
                   Deserialize and load

Training GPU 2,3:  [weights for engine 1]  → similar flow
Training GPU 4,5:  [weights for engine 2]  → similar flow
Training GPU 6,7:  [weights for engine 3]  → similar flow
```

---

# 5. EVALUATION AND CHECKPOINTING

## 5.1 Evaluation

**Function**: `rollout_manager.eval(rollout_id)`
**File**: `/home/user/slime/slime/ray/rollout.py` (Lines 105-116)

```python
def eval(self, rollout_id):
    if self.args.debug_train_only:
        return  # Skip if training only
    
    # Generate evaluation samples
    data = call_rollout_fn(
        self.eval_generate_rollout,  # Different function for eval
        self.args,
        rollout_id,
        self.data_source,
        evaluation=True
    ).data
    
    # Save debug data
    self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=True)
    
    # Compute and log metrics
    metrics = _log_eval_rollout_data(rollout_id, self.args, data)
    
    # Check metrics (if enabled)
    if self._metric_checker is not None:
        self._metric_checker.on_eval(metrics)
```

### Evaluation Configuration

From script (Lines 55-61):

```bash
EVAL_ARGS=(
   --eval-interval 20         # Evaluate every 20 rollouts
   --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 0.7
)
```

### Metric Computation

**Function**: `_log_eval_rollout_data()`
**File**: `/home/user/slime/slime/ray/rollout.py` (lines after 110)

```python
metrics = {
    "pass_rate": compute_pass_rate(data),  # Accuracy on eval set
    "avg_response_len": mean(response_lengths),
    "truncation_rate": sum(truncated) / len(truncated),
}
```

## 5.2 Model Checkpointing

**Function**: `actor_model.save_model(step_id)`
**File**: `/home/user/slime/slime/ray/actor_group.py` (Lines 122-124)

```python
return ray.get([actor.save_model.remote(step_id) for actor in self._actor_handlers])
```

### Save Implementation

**Function**: `MegatronTrainRayActor.save_model(iteration)`
**File**: `/home/user/slime/slime/backends/megatron_utils/actor.py` (Lines 397-401)

```python
def save_model(self, iteration: int) -> None:
    if self.args.debug_rollout_only:
        return
    
    # Megatron's save function
    save(iteration, self.model, self.optimizer, self.opt_param_scheduler)
```

### Megatron Save Process

**Function**: `save()` from Megatron
**File**: `/home/user/slime/slime/backends/megatron_utils/model.py` (lines 550+)

```python
def save(iteration, model, optimizer, opt_param_scheduler):
    # 1. Only save from rank 0 of each TP group
    if mpu.get_tensor_model_parallel_rank() == 0:
        
        # 2. Convert model to HF format
        # 3. Save to --save /root/Qwen3-4B_slime/
        save_checkpoint(
            iteration=iteration,
            model=model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            args=args
        )
```

### Checkpoint Structure

```
/root/Qwen3-4B_slime/
├── iter_0000060_zero/      # After rollout 60
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── tokenizer.model
├── iter_0000120_zero/      # After rollout 120
└── latest_checkpointed_iteration.txt
```

### Save Interval

From script (Lines 29-36):

```bash
CKPT_ARGS=(
   --save-interval 20    # Save every 20 rollouts
   --save /root/Qwen3-4B_slime/
)
```

---

# 6. DATA STRUCTURES AND MESSAGE PASSING

## 6.1 Rollout Data Structure

**Class**: `Sample` (slime/utils/types.py)

```python
@dataclass
class Sample:
    tokens: List[int]           # Token IDs (prompt + response)
    response_length: int        # Length of response portion
    reward: float              # Scalar reward from reward model
    status: Status             # TRUNCATED or SUCCESS
    truncated: int             # 1 if truncated, 0 otherwise
    index: int                 # Original sample index
    loss_mask: Optional[List[int]]  # Mask for response tokens
    metadata: Optional[Dict]   # Additional metadata
```

## 6.2 RolloutBatch (Training Data)

After conversion by `_convert_samples_to_train_data()`:

```python
@dataclass
class RolloutBatch(dict):
    tokens: List[torch.Tensor]           # [1, T] per sample
    response_lengths: List[int]
    rewards: List[float]                 # Normalized
    raw_reward: List[float]              # Unnormalized
    truncated: List[int]                 # 1/0 flags
    sample_indices: List[int]
    loss_masks: List[torch.Tensor]       # Mask per response
    
    # Computed during training
    log_probs: List[torch.Tensor]        # Old policy log probs
    ref_log_probs: List[torch.Tensor]    # Ref model log probs
    values: List[torch.Tensor]           # Value head outputs
    advantages: List[torch.Tensor]       # Computed advantages
    returns: List[torch.Tensor]          # Computed returns
```

## 6.3 Message Passing via Ray

### RolloutManager → Actor

```python
# Train loop
rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
# rollout_data_ref = ObjectRef pointing to RolloutBatch in Ray object store

actor_model.async_train(rollout_id, rollout_data_ref)
# Passes reference, not copy (efficient for large data)
```

### Actor → SGLang Engine

```python
# Weight update
actor_model.update_weights()
# Serializes weights to bytes, sends via Ray IPC
# Each engine receives relevant shards
```

---

# 7. GPU COMMUNICATION SUMMARY

## 7.1 NCCL Communications

### Data Parallel All-Reduce (Loss/Metrics)

```
8 training ranks, DP groups of size [2,2,2,2]:
Group 0: ranks [0,1]  → avg loss
Group 1: ranks [2,3]  → avg loss
Group 2: ranks [4,5]  → avg loss
Group 3: ranks [6,7]  → avg loss
```

### Tensor Parallel All-Gather (Weight Update)

```
TP size = 2, pairs: [0-1], [2-3], [4-5], [6-7]
Each pair all-gathers to create full weight

GPU 0 shard_0 + GPU 1 shard_1 → [full weight on both]
```

## 7.2 Intra-GPU Communication (Colocated)

Training and inference on same 8 GPUs:
- Training uses ranks 0-7
- Inference engines use subsets via torch_memory_saver
- Offloading moves inactive components to CPU via LD_PRELOAD hook

---

# 8. COMPLETE SEQUENCE DIAGRAM

```
Rollout ID: N

1. GENERATE ROLLOUTS
   train.py → RolloutManager.generate()
   ↓
   SGLang engines [on GPUs 0,2,4,6] read weights
   ↓
   Generate 8 samples per prompt, compute rewards
   ↓
   Return: {tokens, rewards, loss_masks, ...}
   Put in Ray object store → ObjectRef

2. PREPARE TRAINING
   offload_rollout() [if enabled]
   ↓
   Move training model weights to GPU (wake_up)
   ↓
   Clear GPU memory

3. TRAIN ACTOR
   For each training rank [0-7]:
   
   a) Fetch rollout data from object store
   b) Create data iterators (dynamic batch sizing)
   c) Run forward pass (with ref model if needed)
   d) Compute log probs and advantages
   e) Run training step:
      - Forward: logits = model(tokens)
      - Compute loss: policy_loss + entropy_loss
      - Backward: loss.backward()
      - All-reduce gradients across DP group
      - Optimizer step (Adam)
   
   All ranks synchronize:
   dist.barrier()

4. UPDATE WEIGHTS
   all_gather TP shards → full weights
   ↓
   convert to HF format
   ↓
   serialize to bytes
   ↓
   gather via Gloo (CPU intermediate)
   ↓
   send to SGLang engines via Ray RPC
   ↓
   engines load weights

5. EVALUATION (every 20 rollouts)
   SGLang engines generate eval samples
   ↓
   Compute pass rate, metrics
   ↓
   Log to W&B

6. CHECKPOINT (every 20 rollouts)
   Rank 0 of each TP group:
   ↓
   save model weights + optimizer state
   ↓
   save to /root/Qwen3-4B_slime/iter_{N:07d}_zero/

7. Next iteration: N+1
```

---

# 9. KEY FILE LOCATIONS AND LINE NUMBERS

## Main Files

| Component | File | Key Functions/Classes |
|-----------|------|---------------------|
| Entry Point | `/home/user/slime/train.py` | `train()` (L14-112) |
| Arg Parsing | `/home/user/slime/slime/utils/arguments.py` | `parse_args()`, `get_slime_extra_args_provider()` |
| Ray Setup | `/home/user/slime/slime/ray/placement_group.py` | `create_placement_groups()` (L71), `create_rollout_manager()` (L160), `create_training_models()` (L123) |
| Actor Group | `/home/user/slime/slime/ray/actor_group.py` | `RayTrainGroup` (L11-149) |
| Rollout Manager | `/home/user/slime/slime/ray/rollout.py` | `RolloutManager` (L35-260) |
| Megatron Actor | `/home/user/slime/slime/backends/megatron_utils/actor.py` | `MegatronTrainRayActor.init()` (L42), `train_actor()` (L304), `update_weights()` (L403) |
| Model Training | `/home/user/slime/slime/backends/megatron_utils/model.py` | `initialize_model_and_optimizer()` (L685), `train_one_step()` (L291), `forward_only()` (L159) |
| Loss Computation | `/home/user/slime/slime/backends/megatron_utils/loss.py` | `compute_advantages_and_returns()` (L192), `policy_loss_function()` (L364) |
| Data Processing | `/home/user/slime/slime/backends/megatron_utils/data.py` | `get_data_iterator()` (L209), `get_batch()` (L22) |
| Weight Update | `/home/user/slime/slime/backends/megatron_utils/update_weight_utils.py` | `UpdateWeightFromTensor` (L334), `all_gather_params_async()` (L76) |
| Initialization | `/home/user/slime/slime/backends/megatron_utils/initialize.py` | `init()` (L52), `_initialize_distributed()` (L29) |

## Shell Scripts

| Script | Location | Purpose |
|--------|----------|---------|
| Main launcher | `/home/user/slime/scripts/run-qwen3-4B.sh` | Orchestrates Ray + training |
| Model config | `/home/user/slime/scripts/models/qwen3-4B.sh` | Model architecture parameters |

