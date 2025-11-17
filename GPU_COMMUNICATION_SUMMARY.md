# GPU Communication Summary - Quick Reference

## Document Location
Full detailed analysis: `/home/user/slime/GPU_COMMUNICATION_ANALYSIS.md`

## Key Findings

### Total Communication Locations Found: 50+

## Communication Types by Frequency

### 1. **All-Reduce Operations** (8 locations)
The most common operation for gradient and loss aggregation.
- **Primary Use**: Gradient reduction, loss synchronization, entropy computation
- **Files**: ppo_utils.py, model.py, data.py, distributed_utils.py, cp_utils.py, fsdp actor.py
- **Key Reduction Ops**: MAX (for synchronizing max values), SUM (default), AVG

### 2. **All-Gather Operations** (4 locations)
Collecting distributed data from all ranks.
- **Primary Use**: Parameter gathering, attention hidden states
- **Files**: hf_attention.py, update_weight_utils.py, fsdp actor.py
- **Specialization**: Async operations (async_op=True) for overlap with computation

### 3. **Broadcast Operations** (5 locations)
Distributing data from one rank to all.
- **Primary Use**: Parameter distribution, weight updates
- **Files**: update_weight_utils.py, data.py, fsdp update_weight_utils.py
- **Pattern**: Often async in batches for efficiency

### 4. **Object Gathering** (5 locations)
Collecting Python objects (not just tensors).
- **Primary Use**: Metadata collection, metrics aggregation
- **Files**: data.py, update_weight_utils.py, fsdp actor.py
- **Backend**: Gloo (CPU-based for efficiency)

### 5. **Barrier Synchronization** (5 locations)
Ensuring all ranks reach synchronization points.
- **Primary Use**: Weight update coordination, checkpoint operations
- **Files**: update_weight_utils.py, actor.py, checkpoint.py
- **Backend**: Gloo (for lightweight CPU synchronization)

### 6. **Custom CP Operations** (2 locations)
Context parallel specific implementations.
- **Primary Use**: Ring attention, sequence gathering
- **Files**: cp_utils.py, hf_attention.py

### 7. **P2P Communication** (3 locations)
Point-to-point send/recv, isend/irecv, P2POp.
- **Primary Use**: Pipeline parallelism
- **Files**: reloadable_process_group.py (monkey patched)
- **Pattern**: Async (isend/irecv) for overlapping with other operations

## Process Groups Used

| Group | Backend | Purpose | Files |
|-------|---------|---------|-------|
| Default (World) | NCCL | GPU tensor operations | All |
| Data Parallel (DP) | NCCL | Gradient/loss sync | Most files |
| Tensor Parallel (TP) | NCCL | Parameter gathering | update_weight_utils.py |
| Pipeline Parallel (PP) | NCCL | Stage sync | megatron files |
| Expert Parallel (EP) | NCCL | MoE communication | update_weight_utils.py |
| Context Parallel (CP) | NCCL | Sequence gathering | cp_utils.py, hf_attention.py |
| Gloo | Gloo | CPU/metadata ops | distributed_utils.py, data.py, checkpoint.py |

## Communication Backends

### NCCL (GPU-to-GPU)
- Used for: Tensor collective operations, parameter gathering, attention computation
- Characteristics: High bandwidth, low latency, GPU memory operations

### Gloo (CPU)
- Used for: Barrier synchronization, object gathering, metadata operations
- Characteristics: CPU-based, supports Python objects, efficient for non-tensor data

## File Organization

### Core Distributed Utils
- `/home/user/slime/slime/utils/distributed_utils.py`: Process group initialization, whitening
- `/home/user/slime/slime/utils/reloadable_process_group.py`: Dynamic PG management

### Megatron Backend
- `/home/user/slime/slime/backends/megatron_utils/update_weight_utils.py`: Weight sync (40+ locations)
- `/home/user/slime/slime/backends/megatron_utils/data.py`: Data sync, metrics gathering
- `/home/user/slime/slime/backends/megatron_utils/model.py`: Training loss reduction
- `/home/user/slime/slime/backends/megatron_utils/cp_utils.py`: Context parallel operations
- `/home/user/slime/slime/backends/megatron_utils/actor.py`: Training barriers

### FSDP Backend
- `/home/user/slime/slime/backends/fsdp_utils/actor.py`: All-reduce, all-gather object
- `/home/user/slime/slime/backends/fsdp_utils/update_weight_utils.py`: Weight gathering
- `/home/user/slime/slime/backends/fsdp_utils/checkpoint.py`: Checkpoint barriers

### Model-Specific
- `/home/user/slime/slime_plugins/models/hf_attention.py`: Attention all-gather
- `/home/user/slime/slime/utils/ppo_utils.py`: Entropy computation all-reduce

## Critical Communication Patterns

### Pattern 1: Parameter Distribution Pipeline
1. Weight update on rank 0
2. All-gather from TP ranks (reconstruct full tensor)
3. Broadcast to PP/EP ranks
4. Scatter to inference engines

### Pattern 2: Gradient Synchronization
1. Local backward pass
2. All-reduce across DP group
3. Weight update

### Pattern 3: Metric Collection
1. Compute local metrics
2. Gather object to DP source rank
3. Aggregate and log

### Pattern 4: Context Parallel Sequence Processing
1. Local forward pass on CP rank's chunk
2. All-gather with ring pattern for attention
3. Reduce/scatter back to local chunks

### Pattern 5: Ring Flash Attention (FSDP)
1. All-gather hidden states across CP group
2. Reconstruct ring pattern (2 chunks per rank)
3. Compute attention on full sequence
4. Scatter back to local chunks

## Async/Overlap Strategies

### All-Gather Async (Parameter Gathering)
- Start async operations for all parameters
- Wait all at once (instead of individually)
- Reduces stalls, enables overlapping with computation

### Broadcast Async (Data/Parameter Sync)
- Multiple broadcasts in a batch
- Collect handles and wait together
- Improves throughput

### Overlapping Communication & Computation
- `no_sync()` during gradient accumulation
- `start_grad_sync()` for overlap control
- Forward pre-hooks for parameter gathering

## Synchronization Points

| Location | Type | Backend | Purpose |
|----------|------|---------|---------|
| Weight update start/end | Barrier | Gloo | Ensure all ready |
| Checkpoint load/save | Barrier | Gloo | File consistency |
| Memory offload/wake | Barrier | Gloo | Memory coordination |
| Convert tool start/end | Barrier | Gloo | Format conversion sync |
| TP gather completion | CUDA sync | - | Memory transfer completion |

## Memory & Efficiency Considerations

1. **Serialization**: Large tensors serialized to bytes for gather_object
2. **Ring Pattern**: CP uses ring structure to reduce memory while gathering
3. **Gloo for Metadata**: Avoids GPU memory pressure for non-tensor operations
4. **Async Operations**: Enable computation-communication overlap
5. **Coalescing**: Multiple operations batched before waiting

## Error Handling

- Memory diagnostics captured on distributed operation failures
- Exception wrapper in reloadable_process_group.py
- Helps debugging communication-related OOMs

