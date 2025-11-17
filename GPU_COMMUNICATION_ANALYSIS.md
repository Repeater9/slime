# GPU Communication Analysis - Comprehensive Report

## Overview
This document comprehensively catalogs all GPU communication patterns in the SLIME codebase. The codebase uses distributed training with multiple parallelism strategies: Data Parallelism (DP), Tensor Parallelism (TP), Pipeline Parallelism (PP), Expert Parallelism (EP), and Context Parallelism (CP). Communication happens via PyTorch distributed operations (NCCL/Gloo backends) and custom collective operations.

---

## 1. COLLECTIVE COMMUNICATION OPERATIONS

### 1.1 All-Reduce Operations
All-reduce aggregates values across ranks by performing a reduction followed by a broadcast.

#### Location 1: PPO Entropy Computation
- **File**: `/home/user/slime/slime/utils/ppo_utils.py`
- **Lines**: 95, 99, 102
- **Function**: `_VocabParallelEntropy.forward()`
- **Type**: Collective (All-Reduce)
- **Communication Details**:
  ```python
  dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=process_group)
  dist.all_reduce(normalized_sum_exp_logits, group=process_group)
  dist.all_reduce(sum_softmax_times_logits, group=process_group)
  ```
- **Purpose**: Aggregates maximum logits, sum of exponentials, and weighted sum across tensor parallel ranks for entropy computation
- **Context**: Part of vocabulary-parallel entropy calculation for reinforcement learning

#### Location 2: Megatron Data Synchronization
- **File**: `/home/user/slime/slime/backends/megatron_utils/data.py`
- **Line**: 266
- **Function**: `get_data_iterator()`
- **Type**: Collective (All-Reduce with MAX operation)
- **Communication Details**:
  ```python
  dist.all_reduce(num_microbatches, op=dist.ReduceOp.MAX, group=dp_group)
  ```
- **Purpose**: Synchronizes the maximum number of microbatches across data parallel ranks for balanced batch processing
- **Context**: Dynamic batch sizing in distributed training

#### Location 3: Megatron Model Loss Reduction
- **File**: `/home/user/slime/slime/backends/megatron_utils/model.py`
- **Line**: 486
- **Function**: `train_one_step()`
- **Type**: Collective (All-Reduce)
- **Communication Details**:
  ```python
  torch.distributed.all_reduce(values, group=mpu.get_data_parallel_group(with_context_parallel=True))
  ```
- **Purpose**: Aggregates loss values across data parallel group (including context parallel) for gradient computation
- **Context**: Loss reduction in multi-GPU training with context parallelism

#### Location 4: Megatron MTP Loss Reduction
- **File**: `/home/user/slime/slime/backends/megatron_utils/model.py`
- **Lines**: 608, 610
- **Function**: `train_one_step()`
- **Type**: Collective (All-Reduce with AVG operation)
- **Communication Details**:
  ```python
  torch.distributed.all_reduce(values, group=tracker.get("reduce_group"))
  torch.distributed.all_reduce(values, group=tracker["avg_group"], op=torch.distributed.ReduceOp.AVG)
  ```
- **Purpose**: Aggregates multi-token prediction losses across ranks with averaging
- **Context**: MTP loss logging and synchronization

#### Location 5: FSDP Model Actor Training
- **File**: `/home/user/slime/slime/backends/fsdp_utils/actor.py`
- **Line**: 367
- **Function**: `train()`
- **Type**: Collective (All-Reduce with MAX operation)
- **Communication Details**:
  ```python
  dist.all_reduce(num_microbatches, op=dist.ReduceOp.MAX, group=self.dp_group)
  ```
- **Purpose**: Synchronizes microbatch count across data parallel ranks in FSDP training
- **Context**: FSDP-based distributed training

#### Location 6: FSDP Metric Aggregation
- **File**: `/home/user/slime/slime/backends/fsdp_utils/actor.py`
- **Line**: 458
- **Function**: `train()`
- **Type**: Collective (All-Reduce with SUM operation)
- **Communication Details**:
  ```python
  dist.all_reduce(val, op=dist.ReduceOp.SUM, group=self.dp_group)
  ```
- **Purpose**: Aggregates training metrics across data parallel ranks
- **Context**: Metric synchronization in FSDP training

#### Location 7: Distributed Whitening
- **File**: `/home/user/slime/slime/utils/distributed_utils.py`
- **Line**: 131
- **Function**: `distributed_masked_whiten()`
- **Type**: Collective (All-Reduce)
- **Communication Details**:
  ```python
  dist.all_reduce(stats_tensor, group=process_group)
  ```
- **Purpose**: Aggregates statistics (sum, sum of squares, mask sum) for whitening normalization
- **Context**: Global normalization using statistics from all GPUs

#### Location 8: Context Parallel All-Reduce
- **File**: `/home/user/slime/slime/backends/megatron_utils/cp_utils.py`
- **Line**: 154
- **Function**: `all_gather_with_cp()`
- **Type**: Collective (All-Reduce)
- **Communication Details**:
  ```python
  full_tensor = dist.nn.all_reduce(full_tensor, group=cp_group)
  ```
- **Purpose**: Aggregates tensor chunks across context parallel ranks after gathering
- **Context**: Context parallel computation for sequence length parallelism

---

### 1.2 All-Gather Operations
All-gather collects data from all ranks and sends the complete collection to all ranks.

#### Location 1: Attention Layer CP
- **File**: `/home/user/slime/slime_plugins/models/hf_attention.py`
- **Lines**: 63-66
- **Function**: `forward()`
- **Type**: Collective (All-Gather)
- **Communication Details**:
  ```python
  hidden_states_list = dist.nn.all_gather(
      hidden_states,
      group=mpu.get_context_parallel_group(),
  )
  ```
- **Purpose**: Gathers hidden states across all context parallel ranks for attention computation
- **Context**: Context parallel attention mechanism

#### Location 2: Megatron TP Parameter Gathering
- **File**: `/home/user/slime/slime/backends/megatron_utils/update_weight_utils.py`
- **Line**: 60
- **Function**: `all_gather_param()`
- **Type**: Collective (All-Gather)
- **Communication Details**:
  ```python
  dist.all_gather(param_partitions, param.data, group=tp_group)
  ```
- **Purpose**: Gathers tensor-parallel sharded parameters to reconstruct full tensor
- **Context**: Weight conversion from distributed format to HuggingFace format

#### Location 3: Megatron Async TP Parameter Gathering
- **File**: `/home/user/slime/slime/backends/megatron_utils/update_weight_utils.py`
- **Line**: 106
- **Function**: `all_gather_params_async()`
- **Type**: Collective (All-Gather Async)
- **Communication Details**:
  ```python
  handle = dist.all_gather(param_partitions, param.data, group=tp_group, async_op=True)
  ```
- **Purpose**: Asynchronously gathers tensor-parallel parameters to enable overlapping with computation
- **Context**: Efficient parameter gathering for weight updates

#### Location 4: FSDP Weight Broadcasting
- **File**: `/home/user/slime/slime/backends/fsdp_utils/actor.py`
- **Line**: 949
- **Function**: `_compute_logits_and_values_with_cp_stacked()`
- **Type**: Collective (All-Gather)
- **Communication Details**:
  ```python
  gathered_stacked = torch.distributed.nn.functional.all_gather(stacked_local, group=cp_group)
  ```
- **Purpose**: Gathers stacked tensors across context parallel group
- **Context**: FSDP-based training with context parallelism

---

### 1.3 All-Gather Object Operations
Gathers Python objects (not just tensors) from all ranks.

#### Location 1: Megatron PP Parameter Info
- **File**: `/home/user/slime/slime/backends/megatron_utils/update_weight_utils.py`
- **Line**: 254
- **Function**: `get_param_infos()`
- **Type**: Collective (All-Gather Object)
- **Communication Details**:
  ```python
  dist.all_gather_object(
      obj=(rank, param_infos), object_list=param_infos_list, group=mpu.get_pipeline_model_parallel_group()
  )
  ```
- **Purpose**: Gathers parameter metadata across pipeline parallel ranks
- **Context**: Cross-rank parameter information synchronization

#### Location 2: Megatron EP Parameter Info
- **File**: `/home/user/slime/slime/backends/megatron_utils/update_weight_utils.py`
- **Line**: 271
- **Function**: `get_param_infos()`
- **Type**: Collective (All-Gather Object)
- **Communication Details**:
  ```python
  dist.all_gather_object(
      obj=(rank, param_infos), object_list=param_infos_list, group=mpu.get_expert_model_parallel_group()
  )
  ```
- **Purpose**: Gathers parameter info across expert parallel ranks for expert parameter handling
- **Context**: Mixture of Experts parameter synchronization

#### Location 3: Global Parameter Info Validation
- **File**: `/home/user/slime/slime/backends/megatron_utils/update_weight_utils.py`
- **Line**: 286
- **Function**: `get_param_infos()`
- **Type**: Collective (All-Gather Object)
- **Communication Details**:
  ```python
  dist.all_gather_object(
      obj=param_infos,
      object_list=all_param_info_list,
      group=get_gloo_group(),
  )
  ```
- **Purpose**: Validates that all ranks have identical parameter information
- **Context**: Cross-rank parameter consistency verification

#### Location 4: Megatron Expert Names
- **File**: `/home/user/slime/slime/backends/megatron_utils/update_weight_utils.py`
- **Line**: 743
- **Function**: `update_weights_from_megatron()`
- **Type**: Collective (All-Gather Object)
- **Communication Details**:
  ```python
  dist.all_gather_object(all_names, names, group=mpu.get_expert_model_parallel_group())
  ```
- **Purpose**: Collects expert parameter names from all expert ranks
- **Context**: Expert parameter collection during weight updates

#### Location 5: FSDP Metric Aggregation via All-Gather
- **File**: `/home/user/slime/slime/backends/fsdp_utils/actor.py`
- **Line**: 642
- **Function**: `train()`
- **Type**: Collective (All-Gather Object)
- **Communication Details**:
  ```python
  dist.all_gather_object(reduced_aggregated, aggregated, group=self.dp_group)
  ```
- **Purpose**: Gathers aggregated metrics from all data parallel ranks
- **Context**: FSDP training metric collection

---

### 1.4 Broadcast Operations
Broadcast sends data from one rank to all other ranks.

#### Location 1: Megatron PP Parameter Broadcast
- **File**: `/home/user/slime/slime/backends/megatron_utils/update_weight_utils.py`
- **Line**: 467
- **Function**: `_gather_bucket_params()`
- **Type**: Collective (Broadcast)
- **Communication Details**:
  ```python
  torch.distributed.broadcast(
      param, src=info.src_rank, group=mpu.get_pipeline_model_parallel_group(), async_op=True
  )
  ```
- **Purpose**: Broadcasts parameters across pipeline parallel ranks
- **Context**: Parameter synchronization in PP training

#### Location 2: Megatron EP Parameter Broadcast
- **File**: `/home/user/slime/slime/backends/megatron_utils/update_weight_utils.py`
- **Line**: 484
- **Function**: `_gather_bucket_params()`
- **Type**: Collective (Broadcast)
- **Communication Details**:
  ```python
  torch.distributed.broadcast(
      param, src=src_rank, group=mpu.get_expert_model_parallel_group(), async_op=True
  )
  ```
- **Purpose**: Broadcasts expert parameters across expert parallel ranks
- **Context**: Expert parameter synchronization

#### Location 3: Megatron Distributed Weight Broadcast
- **File**: `/home/user/slime/slime/backends/megatron_utils/update_weight_utils.py`
- **Line**: 865
- **Function**: `update_weights_from_distributed()`
- **Type**: Collective (Broadcast Async)
- **Communication Details**:
  ```python
  handles.append(dist.broadcast(param.data, 0, group=group, async_op=True))
  ```
- **Purpose**: Broadcasts updated parameters to rollout engines
- **Context**: Distributed weight synchronization for inference engines

#### Location 4: FSDP Weight Broadcast
- **File**: `/home/user/slime/slime/backends/fsdp_utils/update_weight_utils.py`
- **Line**: 421
- **Function**: `_broadcast_params()`
- **Type**: Collective (Broadcast)
- **Communication Details**:
  ```python
  dist.broadcast(param_data, 0, group=self._model_update_groups, async_op=False)
  ```
- **Purpose**: Broadcasts model parameters from rank 0 to all inference engines
- **Context**: FSDP model parameter distribution

#### Location 5: Megatron Data Broadcasting
- **File**: `/home/user/slime/slime/backends/megatron_utils/data.py`
- **Lines**: 465, 472-473
- **Function**: `sync_rollout_data()`
- **Type**: Collective (Broadcast Async)
- **Communication Details**:
  ```python
  handles.append(dist.broadcast(value, src=1, group=group, async_op=True))
  handles.append(dist.broadcast(ref_log_prob, src=0, group=group, async_op=True))
  handles.append(dist.broadcast(log_prob, src=0, group=group, async_op=True))
  ```
- **Purpose**: Broadcasts values, log probabilities, and reference log probabilities
- **Context**: Rollout data synchronization across pipeline stages

---

### 1.5 Gather Object Operations
Gathers Python objects from all ranks to a single destination rank.

#### Location 1: Megatron Data Metric Gathering
- **File**: `/home/user/slime/slime/backends/megatron_utils/data.py`
- **Lines**: 106, 136
- **Function**: `gather_log_data()`
- **Type**: Collective (Gather Object)
- **Communication Details**:
  ```python
  dist.gather_object(
      log_dict,
      gathered_log_dict,
      dst=mpu.get_data_parallel_src_rank(with_context_parallel=True),
      group=mpu.get_data_parallel_group_gloo(with_context_parallel=True),
  )
  ```
- **Purpose**: Gathers per-rank metrics to DP source rank for logging
- **Context**: Distributed metric collection using Gloo backend (CPU communication)

#### Location 2: Megatron Weight Update Serialized Gathering
- **File**: `/home/user/slime/slime/backends/megatron_utils/update_weight_utils.py`
- **Line**: 555
- **Function**: `_send_to_colocated_engine()`
- **Type**: Collective (Gather Object via Gloo)
- **Communication Details**:
  ```python
  dist.gather_object(
      serialized_tensor_data,
      gathered_data,
      dst=self._ipc_gather_src,
      group=self._ipc_gather_group,
  )
  ```
- **Purpose**: Gathers serialized tensors via Gloo for efficient CPU-based aggregation
- **Context**: Colocated engine weight distribution

#### Location 3: FSDP Weight Update Gathering
- **File**: `/home/user/slime/slime/backends/fsdp_utils/update_weight_utils.py`
- **Lines**: 190, 261
- **Function**: `_broadcast_params()`, `_broadcast_params_no_serialize()`
- **Type**: Collective (Gather Object via Gloo)
- **Communication Details**:
  ```python
  dist.gather_object(
      serialized_data,
      gathered_data,
      dst=0,
      group=self._model_update_groups,
  )
  ```
- **Purpose**: Gathers serialized parameters from all ranks to rank 0 for weight updates
- **Context**: FSDP-based distributed weight gathering

---

### 1.6 Barrier Synchronization
Barriers ensure all ranks reach a synchronization point before proceeding.

#### Location 1: Weight Update Synchronization
- **File**: `/home/user/slime/slime/backends/megatron_utils/update_weight_utils.py`
- **Lines**: 425, 435
- **Function**: `update_weights()`
- **Type**: Synchronization (Barrier)
- **Communication Details**:
  ```python
  dist.barrier(group=get_gloo_group())
  ```
- **Purpose**: Synchronizes all ranks before and after weight updates
- **Context**: Gloo-based barrier for weight synchronization

#### Location 2: Megatron Actor Barrier
- **File**: `/home/user/slime/slime/backends/megatron_utils/actor.py`
- **Lines**: 65, 89, 241, 260, 416, 683
- **Function**: Multiple locations in training loop
- **Type**: Synchronization (Barrier)
- **Communication Details**:
  ```python
  dist.barrier(group=get_gloo_group())
  ```
- **Purpose**: Ensures synchronization at critical points in training
- **Context**: Megatron distributed training synchronization

#### Location 3: FSDP Offload/Wake Barrier
- **File**: `/home/user/slime/slime/backends/fsdp_utils/actor.py`
- **Lines**: 241, 260, 683
- **Function**: `sleep()`, `wake_up()`, `update_weights()`
- **Type**: Synchronization (Barrier)
- **Communication Details**:
  ```python
  dist.barrier(group=get_gloo_group())
  ```
- **Purpose**: Synchronizes memory management operations in FSDP
- **Context**: Memory offloading and restoration coordination

#### Location 4: FSDP Checkpoint Barrier
- **File**: `/home/user/slime/slime/backends/fsdp_utils/checkpoint.py`
- **Lines**: 137, 160, 181, 214
- **Function**: Checkpoint load/save operations
- **Type**: Synchronization (Barrier)
- **Communication Details**:
  ```python
  dist.barrier()
  ```
- **Purpose**: Coordinates checkpoint loading and saving across ranks
- **Context**: Distributed checkpoint operations

#### Location 5: Convert Tools Barrier
- **File**: `/home/user/slime/tools/convert_to_hf.py`, `/home/user/slime/tools/convert_hf_to_torch_dist.py`
- **Line**: 108
- **Function**: Model conversion
- **Type**: Synchronization (Barrier)
- **Communication Details**:
  ```python
  dist.barrier()
  ```
- **Purpose**: Ensures all ranks complete conversion before proceeding
- **Context**: Model format conversion

---

### 1.7 Reduce Operations
Reduce aggregates values across ranks and leaves result only on destination rank.

#### Location 1: Reloadable Process Group Reduce
- **File**: `/home/user/slime/slime/utils/reloadable_process_group.py`
- **Line**: 70-71, 186
- **Function**: Monkey patched torch.distributed
- **Type**: Collective (Reduce)
- **Communication Details**:
  ```python
  dist.reduce = get_new_function(dist.reduce)
  ```
- **Purpose**: Reduces values on a single destination rank
- **Context**: Generic reduce operation for process groups

---

### 1.8 Reduce-Scatter Operations
Reduce-scatter combines reduction with scatter, reducing values and scattering results.

#### Location 1: Reloadable Process Group Reduce-Scatter
- **File**: `/home/user/slime/slime/utils/reloadable_process_group.py`
- **Lines**: 71-72, 207-214
- **Function**: Monkey patched torch.distributed
- **Type**: Collective (Reduce-Scatter)
- **Communication Details**:
  ```python
  dist.reduce_scatter = get_new_function(dist.reduce_scatter)
  dist.reduce_scatter_tensor = get_new_function(dist.reduce_scatter_tensor)
  ```
- **Purpose**: Generic reduce-scatter operations for process groups
- **Context**: Gradient reduction and scatter in distributed optimization

---

### 1.9 Scatter and Gather Operations
Scatter distributes data from one rank to all. Gather collects data from all to one rank.

#### Location 1: Reloadable Process Group Scatter/Gather
- **File**: `/home/user/slime/slime/utils/reloadable_process_group.py`
- **Lines**: 73-74, 201-205
- **Function**: Monkey patched torch.distributed
- **Type**: Collective (Scatter/Gather)
- **Communication Details**:
  ```python
  dist.scatter = get_new_function(dist.scatter)
  dist.gather = get_new_function(dist.gather)
  ```
- **Purpose**: Generic scatter and gather operations
- **Context**: Data distribution and collection in distributed training

---

### 1.10 All-to-All Operations
All-to-all exchanges data where each rank sends different data to every other rank.

#### Location 1: Reloadable Process Group All-to-All
- **File**: `/home/user/slime/slime/utils/reloadable_process_group.py`
- **Lines**: 67-68
- **Function**: Monkey patched torch.distributed
- **Type**: Collective (All-to-All)
- **Communication Details**:
  ```python
  dist.all_to_all = get_new_function(dist.all_to_all)
  dist.all_to_all_single = get_new_function(dist.all_to_all_single)
  ```
- **Purpose**: Generic all-to-all operations for process groups
- **Context**: Expert routing and token distribution in MoE models

---

## 2. POINT-TO-POINT (P2P) COMMUNICATION

### 2.1 Send/Recv Operations
Direct point-to-point communication between pairs of ranks.

#### Location 1: Reloadable Process Group Send/Recv
- **File**: `/home/user/slime/slime/utils/reloadable_process_group.py`
- **Lines**: 76-77, 222-226
- **Function**: Monkey patched torch.distributed
- **Type**: P2P (Send/Recv)
- **Communication Details**:
  ```python
  dist.send = get_new_function(dist.send)
  dist.recv = get_new_function(dist.recv)
  
  def send(self, *a, **kw):
      return self._fwd("send", *a, **kw)
  
  def recv(self, *a, **kw):
      return self._fwd("recv", *a, **kw)
  ```
- **Purpose**: Generic synchronous point-to-point communication
- **Context**: Pipeline parallelism and point-to-point messaging

### 2.2 Isend/Irecv Operations (Async P2P)
Asynchronous point-to-point communication for overlapping computation and communication.

#### Location 1: Reloadable Process Group Isend/Irecv
- **File**: `/home/user/slime/slime/utils/reloadable_process_group.py`
- **Lines**: 81-85, 222-226
- **Function**: Monkey patched torch.distributed
- **Type**: P2P (Async Send/Recv)
- **Communication Details**:
  ```python
  old_isend = dist.isend
  old_irecv = dist.irecv
  
  dist.isend = get_new_function(dist.isend)
  dist.irecv = get_new_function(dist.irecv)
  ```
- **Purpose**: Non-blocking point-to-point communication for pipeline parallelism
- **Context**: Asynchronous send/receive operations in distributed training

### 2.3 P2P Operations via P2POp
Batched point-to-point operations for efficient pipeline communication.

#### Location 1: Reloadable Process Group P2POp
- **File**: `/home/user/slime/slime/utils/reloadable_process_group.py`
- **Lines**: 87-105
- **Function**: `get_new_p2pop_function()`
- **Type**: P2P (Batched Operations)
- **Communication Details**:
  ```python
  dist.P2POp.__new__ = get_new_p2pop_function(dist.P2POp.__new__)
  dist.P2POp.__init__ = get_new_p2pop_function(dist.P2POp.__init__)
  ```
- **Purpose**: Batches multiple P2P operations for efficient communication
- **Context**: Pipeline parallelism with grouped sends/receives

---

## 3. CUSTOM COMMUNICATION IMPLEMENTATIONS

### 3.1 Context Parallel (CP) All-Gather
Custom all-gather implementation optimized for context parallelism.

#### Location 1: CP All-Gather with Ring Pattern
- **File**: `/home/user/slime/slime/backends/megatron_utils/cp_utils.py`
- **Lines**: 106-155
- **Function**: `all_gather_with_cp()`
- **Type**: Custom Collective (All-Gather)
- **Communication Details**:
  ```python
  cp_group = mpu.get_context_parallel_group()
  cp_size = mpu.get_context_parallel_world_size()
  # Gathers tensors from 2 chunks per rank across CP group
  full_tensor = dist.nn.all_reduce(full_tensor, group=cp_group)
  ```
- **Purpose**: Gathers sequence chunks across context parallel ranks with special ring structure handling
- **Context**: Context parallelism for long sequence processing

#### Location 2: CP Token Slicing
- **File**: `/home/user/slime/slime/backends/megatron_utils/cp_utils.py`
- **Lines**: 158-174
- **Function**: `slice_with_cp()`
- **Type**: Custom P2P (Local Slicing)
- **Communication Details**:
  ```python
  cp_rank = mpu.get_context_parallel_rank()
  cp_size = mpu.get_context_parallel_world_size()
  chunk_size = (len(tokens) + 2 * cp_size - 1) // (2 * cp_size)
  # Extracts 2 chunks for this rank
  return torch.cat([tokens[start_1:end_1], tokens[start_2:end_2]])
  ```
- **Purpose**: Distributes token sequences across context parallel ranks
- **Context**: Token distribution in CP training

---

### 3.2 Ring Flash Attention
Custom ring-based attention using all-gather for context parallelism.

#### Location 1: Ring Flash Attention Setup
- **File**: `/home/user/slime/slime/backends/fsdp_utils/actor.py`
- **Lines**: 210-211
- **Function**: `setup_device_mesh()`
- **Type**: Custom Collective (Ring Pattern)
- **Communication Details**:
  ```python
  substitute_hf_flash_attn(self.cp_group, heads_k_stride=1)
  update_ring_flash_attn_params(cu_seqlens, self.cp_group)
  ```
- **Purpose**: Enables ring flash attention for sequence-parallel computation
- **Context**: Memory-efficient attention with context parallelism

#### Location 2: Ring Attention in Attention Layer
- **File**: `/home/user/slime/slime_plugins/models/hf_attention.py`
- **Lines**: 61-85
- **Function**: `forward()`
- **Type**: Custom Collective (All-Gather for Ring Attention)
- **Communication Details**:
  ```python
  hidden_states_list = dist.nn.all_gather(
      hidden_states,
      group=mpu.get_context_parallel_group(),
  )
  # Reconstructs full sequence from CP chunks
  full_tensor = torch.cat([left, chunk_0, mid, chunk_1, right], dim=0)
  ```
- **Purpose**: All-gathers hidden states for ring attention pattern
- **Context**: Context parallel attention computation

---

## 4. PROCESS GROUP MANAGEMENT & INITIALIZATION

### 4.1 Process Group Creation
Creates communication groups for different parallelism strategies.

#### Location 1: Gloo Group Initialization
- **File**: `/home/user/slime/slime/utils/distributed_utils.py`
- **Lines**: 20-34
- **Function**: `init_gloo_group()`, `get_gloo_group()`
- **Type**: Process Group Management
- **Communication Details**:
  ```python
  GLOO_GROUP = dist.new_group(backend="gloo")
  ```
- **Purpose**: Creates a Gloo group for CPU-based collectives (synchronization, metadata)
- **Context**: Efficient CPU communication for collective metadata operations

#### Location 2: Megatron TP Gather Group
- **File**: `/home/user/slime/slime/backends/megatron_utils/update_weight_utils.py`
- **Line**: 368
- **Function**: `UpdateWeightFromTensor.__init__()`
- **Type**: Process Group Management
- **Communication Details**:
  ```python
  new_group = dist.new_group(ranks=group_ranks, backend="gloo")
  self._ipc_gather_group = new_group
  ```
- **Purpose**: Creates Gloo group for intra-engine parameter gathering
- **Context**: Colocated engine weight distribution

#### Location 3: Model Update Groups
- **File**: `/home/user/slime/slime/backends/megatron_utils/update_weight_utils.py`
- **Line**: 821
- **Function**: `connect_rollout_engines_from_distributed()`
- **Type**: Process Group Management
- **Communication Details**:
  ```python
  model_update_groups = init_process_group(
      backend=distributed_config.backend,
      ranks=ranks,
      group_name=f"{group_name}_{i}",
  )
  ```
- **Purpose**: Creates NCCL groups for distributed weight updates
- **Context**: Distributed engine weight synchronization

#### Location 4: FSDP Model Update Groups
- **File**: `/home/user/slime/slime/backends/fsdp_utils/update_weight_utils.py`
- **Line**: 141-147
- **Function**: `__init__()`
- **Type**: Process Group Management
- **Communication Details**:
  ```python
  new_group = dist.new_group(
      ranks=group_ranks,
      backend="gloo"
  )
  self._ipc_gather_group = new_group
  ```
- **Purpose**: Creates Gloo group for FSDP weight gathering
- **Context**: FSDP-based distributed weight management

### 4.2 Device Mesh Setup (FSDP)
Creates multi-dimensional device meshes for parallelism management.

#### Location 1: FSDP Device Mesh
- **File**: `/home/user/slime/slime/backends/fsdp_utils/actor.py`
- **Lines**: 189-196
- **Function**: `setup_device_mesh()`
- **Type**: Process Group Management
- **Communication Details**:
  ```python
  self.mesh = init_device_mesh("cuda", mesh_shape=(self.dp_size, self.cp_size), mesh_dim_names=("dp", "cp"))
  self.dp_group = self.mesh.get_group("dp")
  self.cp_group = self.mesh.get_group("cp")
  self.dp_mesh = self.mesh["dp"]
  ```
- **Purpose**: Creates 2D device mesh for data and context parallelism
- **Context**: Structured parallelism management in FSDP

---

### 4.3 Reloadable Process Groups
Dynamic process group management for checkpoint reload scenarios.

#### Location 1: Reloadable Process Group Wrapper
- **File**: `/home/user/slime/slime/utils/reloadable_process_group.py`
- **Lines**: 12-45
- **Function**: `monkey_patch_torch_dist()`
- **Type**: Process Group Management
- **Communication Details**:
  ```python
  def new_group(*args, **kwargs):
      group = old_new_group(*args, **kwargs)
      group = ReloadableProcessGroup(group, ranks)
      return group
  ```
- **Purpose**: Wraps process groups to allow destruction/recreation on checkpoint reload
- **Context**: Megatron distributed training with checkpointing

#### Location 2: Destroy and Reload Process Groups
- **File**: `/home/user/slime/slime/utils/reloadable_process_group.py`
- **Lines**: 129-148
- **Function**: `ReloadableProcessGroup.destroy_process_groups()`, `reload_process_groups()`
- **Type**: Process Group Management
- **Communication Details**:
  ```python
  dist.destroy_process_group(reloadable_group.group)
  group = old_new_group(ranks=reloadable_group.group_info["ranks"], backend="nccl")
  ```
- **Purpose**: Dynamically destroys and recreates process groups for checkpoint handling
- **Context**: Megatron checkpointing with process group recreation

---

## 5. SYNCHRONIZATION & CUDA OPERATIONS

### 5.1 CUDA Synchronization
Ensures GPU operations complete before proceeding.

#### Location 1: CUDA Synchronize in Tensor Backuper
- **File**: `/home/user/slime/slime/utils/tensor_backper.py`
- **Lines**: 57, 70, 93, 98
- **Function**: Memory management methods
- **Type**: Synchronization (CUDA)
- **Communication Details**:
  ```python
  torch.cuda.synchronize()
  ```
- **Purpose**: Ensures tensor operations complete before memory operations
- **Context**: Memory efficiency operations

#### Location 2: Update Weight Utils Synchronize
- **File**: `/home/user/slime/slime/backends/megatron_utils/update_weight_utils.py`
- **Line**: 458
- **Function**: `_gather_bucket_params()`
- **Type**: Synchronization (CUDA)
- **Communication Details**:
  ```python
  torch.cuda.synchronize()
  ```
- **Purpose**: Ensures memory transfer completes before gather operations
- **Context**: Parameter gathering synchronization

#### Location 3: FSDP Checkpoint Synchronize
- **File**: `/home/user/slime/slime/backends/fsdp_utils/checkpoint.py`
- **Lines**: 159, 169
- **Function**: Checkpoint operations
- **Type**: Synchronization (CUDA)
- **Communication Details**:
  ```python
  torch.cuda.synchronize()
  ```
- **Purpose**: Ensures checkpoint I/O completes before proceeding
- **Context**: Checkpoint synchronization

---

## 6. COMMUNICATION PATTERNS BY PARALLELISM STRATEGY

### 6.1 Data Parallelism (DP)
- **All-Reduce**: Gradient aggregation, loss reduction
- **All-Gather Object**: Metric collection to rank 0
- **Barrier**: Synchronization before/after weight updates
- **Process Group**: `mpu.get_data_parallel_group()`, `mpu.get_data_parallel_group_gloo()`

### 6.2 Tensor Parallelism (TP)
- **All-Gather**: Parameter gathering for full tensor reconstruction
- **All-Gather Async**: Non-blocking parameter gathering for overlap
- **Broadcast**: Parameter distribution across TP ranks
- **Process Group**: `mpu.get_tensor_model_parallel_group()`, `mpu.get_expert_tensor_model_parallel_group()`

### 6.3 Pipeline Parallelism (PP)
- **Broadcast**: Parameter synchronization across PP stages
- **All-Gather Object**: Parameter metadata collection
- **P2P/Isend/Irecv**: Activation and gradient communication between stages
- **Process Group**: `mpu.get_pipeline_model_parallel_group()`

### 6.4 Context Parallelism (CP)
- **All-Gather**: Ring flash attention for sequence gathering
- **All-Reduce**: Context-wide aggregation
- **Custom All-Gather with CP**: Specialized CP token gathering
- **Process Group**: `mpu.get_context_parallel_group()`

### 6.5 Expert Parallelism (EP)
- **All-Gather Object**: Expert name collection, metadata sync
- **Broadcast**: Expert parameter distribution
- **All-Gather**: Expert parameter gathering
- **Process Group**: `mpu.get_expert_model_parallel_group()`

### 6.6 FSDP-Specific Communication
- **All-Reduce**: Gradient aggregation (handled internally by FSDP)
- **Broadcast**: Parameter updates to all ranks
- **All-Gather Object**: Metric aggregation
- **Device Mesh**: 2D mesh for DP and CP coordination
- **Process Groups**: `mesh.get_group("dp")`, `mesh.get_group("cp")`

---

## 7. ASYNC vs SYNC COMMUNICATION

### 7.1 Asynchronous Operations
Operations that return a handle and can be waited on later.

#### All-Gather Async (TP Parameters)
- **File**: `/home/user/slime/slime/backends/megatron_utils/update_weight_utils.py`
- **Line**: 106
- **Usage**: `handle = dist.all_gather(..., async_op=True)` followed by `handle.wait()`

#### Broadcast Async (Multiple Locations)
- **Files**: Various update_weight_utils, data.py
- **Usage**: `handles.append(dist.broadcast(..., async_op=True))` in loops, then `handle.wait()`

### 7.2 Synchronous Operations
Operations that block until completion.

#### All-Reduce
- Default behavior, blocks until all ranks complete
- Used for critical synchronization points

#### Barrier
- Always synchronous, waits for all ranks
- Used for explicit synchronization points

#### Broadcast (Megatron Data)
- Mix of async and sync depending on context

---

## 8. COMMUNICATION BACKENDS

### 8.1 NCCL (GPU Communication)
Used for GPU-to-GPU communication, all collective operations on GPU tensors.

- **Default backend** for distributed operations
- **Used for**:
  - All-reduce, all-gather on GPU tensors
  - Broadcast of tensors
  - P2P operations between GPUs
  - Ring attention and context parallel operations

### 8.2 Gloo (CPU Communication)
Used for CPU-based communication, typically for metadata and synchronization.

- **Process Groups**: `get_gloo_group()`, explicit group creation with `backend="gloo"`
- **Used for**:
  - Barrier operations
  - All-gather object (Python objects)
  - Gather object (metadata, metrics)
  - Inter-node synchronization

### 8.3 Mixed Backend Strategy
- **GPU operations**: NCCL (high bandwidth, low latency)
- **Metadata/sync**: Gloo (supports CPU tensors and Python objects)
- **Process groups**: Can specify backend when creating groups

---

## 9. COMMUNICATION PATTERNS BY DATA TYPE

### 9.1 Tensor Communication
- **Type**: GPU tensors (NCCL)
- **Operations**: All-reduce, all-gather, broadcast
- **Data**: Model parameters, gradients, hidden states, logits

### 9.2 Object Communication
- **Type**: Python objects (serialized)
- **Operations**: All-gather object, gather object
- **Data**: Parameter metadata, metrics, configuration

### 9.3 Mixed Communication
Some operations involve both:
- Serialize tensors to bytes, gather via Gloo, deserialize
- Examples: Weight update gathering in colocated engines

---

## 10. MEMORY & PERFORMANCE CONSIDERATIONS

### 10.1 Async Operations for Overlapping
- Parameter gathering uses `async_op=True` to overlap with computation
- Multiple async operations can be batched and waited together
- Example: `all_gather_params_async()` reduces stalls

### 10.2 Gloo for Metadata
- Gloo operations on CPU to avoid GPU memory overhead
- Metadata operations don't benefit from GPU bandwidth
- Barrier operations use Gloo for efficiency

### 10.3 Ring Attention for Memory Efficiency
- Ring flash attention reduces memory by computing attention in chunks
- Uses all-gather to reconstruct sequences per stage
- Trades communication for memory savings

### 10.4 Serialization for Large Objects
- Complex objects serialized to bytes before gather
- Reduces memory fragmentation and communication overhead
- Used for weight updates in distributed training

---

## 11. ERROR HANDLING & MONITORING

### 11.1 Memory Info on Error
- **File**: `/home/user/slime/slime/utils/reloadable_process_group.py`
- **Line**: 266-272
- **Purpose**: Capture memory information when distributed operations fail
- **Context**: Debugging communication errors

### 11.2 Synchronization with Exception Wrapper
- `_wrap_low_level_call()` context manager
- Adds memory diagnostics to exceptions
- Helps identify memory-related communication failures

---

## SUMMARY TABLE

| Operation Type | Backend | Main Use Case | Key Files |
|---|---|---|---|
| All-Reduce | NCCL | Gradient aggregation, loss reduction | ppo_utils.py, model.py, data.py |
| All-Gather | NCCL | Parameter gathering, attention | update_weight_utils.py, hf_attention.py |
| All-Gather Object | Gloo | Metadata collection | update_weight_utils.py, data.py |
| Broadcast | NCCL/Gloo | Parameter distribution | update_weight_utils.py, data.py |
| Gather Object | Gloo | Metric collection to rank 0 | data.py, actor.py |
| Barrier | Gloo | Synchronization | Multiple files |
| P2P Send/Recv | NCCL | Pipeline communication | reloadable_process_group.py |
| Custom CP All-Gather | NCCL | Context parallel sequences | cp_utils.py |
| Ring Attention | NCCL | Memory-efficient attention | hf_attention.py |
| CUDA Sync | - | GPU operation completion | tensor_backper.py, checkpoint.py |

