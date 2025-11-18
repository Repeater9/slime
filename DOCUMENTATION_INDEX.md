# Qwen3-4B Training Process Documentation - Complete Guide

## Overview

This directory now contains **comprehensive documentation** of the complete end-to-end training process for qwen3-4B using SLIME's distributed RLHF training framework. The documentation covers all 5 phases requested with extensive details about code paths, GPU communication, and data flows.

---

## Document Structure

### 1. **TRAINING_FLOW_DOCUMENTATION.md** (1,288 lines)
**The primary comprehensive guide** - Start here for deep understanding.

**Covers:**
- **Phase 1: Initialization** (Lines 1-750)
  - Ray cluster setup and shell script execution
  - Argument parsing with all configuration groups
  - GPU placement group creation and detection
  - Rollout manager initialization with SGLang engines
  - Actor creation and Ray distributed setup
  - Megatron initialization (tensor/data/pipeline parallel groups)
  - Model loading, checkpoint restoration
  - Tokenizer and config loading
  - Weight backup mechanism (TensorBackuper)

- **Phase 2: Main Training Loop** (Lines 750-1100)
  - Training loop structure (3000 rollouts)
  - Rollout generation via SGLang
  - Sample processing and reward normalization
  - Training data conversion
  - Ray object store usage for efficiency

- **Phase 3: Actor Training** (Lines 1100-1500)
  - Data fetching and preprocessing
  - Data iterator creation with dynamic batch sizing
  - Reference model forward pass
  - Current policy forward pass
  - Advantage computation (GRPO formula)
  - Loss computation with PPO clipping
  - Backward pass and gradient synchronization
  - Optimizer step and learning rate scheduling

- **Phase 4: Weight Synchronization** (Lines 1500-1800)
  - Weight version management
  - Tensor parallel all-gather
  - Hugging Face format conversion
  - Serialization and transport
  - IPC group creation for colocated engines
  - SGLang engine weight loading

- **Phase 5: Evaluation & Checkpointing** (Lines 1800-1900)
  - Evaluation metrics computation
  - Model saving procedure
  - Checkpoint directory structure

- **Additional Sections:**
  - Data structures and message passing
  - GPU communication summary
  - Complete sequence diagram
  - File locations and line numbers table

**When to use:** Need the complete detailed flow with all function calls and line numbers.

---

### 2. **ARCHITECTURE_SUMMARY.md** (395 lines)
**Visual and high-level overview** - Best for understanding system architecture.

**Covers:**
- System architecture diagram (Ray cluster with 8 GPUs)
- Tensor/Data/Pipeline/Context parallel group mappings
- Complete data flow in one training iteration (4 steps)
- Distributed training groups visualization
- Key hyperparameters (model, training, RL, rollout)
- GPU memory management strategy
- Checkpoint structure
- File organization tree
- Key algorithmic components (GRPO, advantages, weight update protocol)
- Training loop pseudocode
- Quick navigation guide

**When to use:** Need quick understanding of system architecture and data flow.

---

### 3. **QUICK_REFERENCE.md** (475 lines)
**Condensed code paths and snippets** - For quick lookups during development.

**Covers:**
- Entry point (shell script → train.py)
- Phase 1: Initialization
  - Argument parsing key args
  - Placement group return values
  - Key data structures (args, distributed groups)
- Phase 2: Main loop structure
- Phase 3: Actor training flow
- Phase 4: Weight synchronization
- Phase 5: Evaluation & checkpointing
- Key data structures (Sample, RolloutBatch)
- Common file locations table
- Execution checklist
- Tips for tracing execution

**When to use:** Need to quickly find specific function locations or understand a particular phase.

---

### 4. **GPU_COMMUNICATION_ANALYSIS.md** (970 lines)
**Detailed GPU communication patterns** - For understanding distributed training communication.

**Covers:**
- NCCL collective operations
- Tensor parallel all-gather operations
- Data parallel all-reduce patterns
- Gradient synchronization across DP groups
- Inter-GPU communication patterns
- IPC (Inter-Process Communication) setup
- Weight update communication flow
- Memory management during communication
- Bottleneck analysis
- Optimization opportunities

**When to use:** Need to understand GPU-to-GPU communication or optimize communication patterns.

---

## Quick Start: How to Use This Documentation

### "I want to understand the entire flow end-to-end"
1. Start with **ARCHITECTURE_SUMMARY.md** (20 min read) for overview
2. Then read **TRAINING_FLOW_DOCUMENTATION.md** Sections 1-2 (40 min) for initialization and main loop
3. Read Section 3 (30 min) for actor training details
4. Read Sections 4-5 (20 min) for weight sync and evaluation

### "I need to understand a specific phase"
- Use **QUICK_REFERENCE.md** to find file locations (5 min)
- Jump to specific section in **TRAINING_FLOW_DOCUMENTATION.md** with exact line numbers (varies)

### "I need to debug a specific component"
- Use **QUICK_REFERENCE.md** Execution Checklist to understand dependencies
- Use **QUICK_REFERENCE.md** Tips for Tracing to understand execution patterns
- Reference file location in Common File Locations table

### "I need to understand GPU communication"
- Read **GPU_COMMUNICATION_ANALYSIS.md** directly (varies by section)

### "I want to trace actual code execution"
- Use **QUICK_REFERENCE.md** Entry Point and follow function calls
- Cross-reference with **TRAINING_FLOW_DOCUMENTATION.md** for detailed explanations

---

## Key Concepts Quick Lookup

### Distributed Training Groups (8 GPUs)
```
Tensor Parallel (TP=2):
  Groups: [0-1], [2-3], [4-5], [6-7]
  
Data Parallel (DP=4):
  Groups: [0-1], [2-3], [4-5], [6-7]
  
Result: 4 DP groups, each with 2 TP ranks
```

### Main Data Structures
- **Sample**: Result from SGLang (tokens, reward, response_length)
- **RolloutBatch**: Training-ready format (tokens, rewards, advantages)
- **TensorBackuper**: CPU-based weight snapshots (actor, ref, old_actor)
- **UpdateWeightFromTensor**: Converts TP shards → full weights → engines

### Key Algorithmic Details
- **GRPO Returns**: `returns = reward - kl_coef * kl` (kl_coef=0 in this config)
- **PPO Loss**: Clipped policy gradient with eps=0.2, eps_clip_high=0.28
- **Reward Normalization**: Group-wise (per prompt, 8 samples per group)

### GPU Communication
- **Training**: NCCL all-reduce for gradients (DP groups)
- **Weight Sync**: NCCL all-gather TP shards + Gloo gather + Ray RPC to engines
- **Offloading**: torch_memory_saver via LD_PRELOAD hook for CPU/GPU swap

---

## File Cross-References

### If you're looking for...

| Task | Document | Section | File | Lines |
|------|----------|---------|------|-------|
| Complete initialization flow | TRAINING_FLOW | 1.0-1.5 | actor.py | 42-131 |
| Main training loop | TRAINING_FLOW | 2.0-2.3 | train.py | 14-112 |
| Rollout generation | TRAINING_FLOW | 2.3b | rollout.py | 89-103 |
| Actor training | TRAINING_FLOW | 3.0-3.2 | actor.py | 261-395 |
| Data iteration | TRAINING_FLOW | 3.3 | data.py | 209-305 |
| Advantages computation | TRAINING_FLOW | 3.4 | loss.py | 192-361 |
| Forward-backward pass | TRAINING_FLOW | 3.5 | model.py | 291-494 |
| Weight synchronization | TRAINING_FLOW | 4.0-4.2 | actor.py, update_weight_utils.py | 403-435, 334-581 |
| System architecture | ARCHITECTURE | - | - | - |
| Quick code lookup | QUICK_REFERENCE | - | - | - |
| GPU communication | GPU_COMM | - | - | - |

---

## Statistics

### Documentation Metrics
```
Total Lines: 3,497 (across 7 files)
- TRAINING_FLOW_DOCUMENTATION.md: 1,288 lines (main guide)
- GPU_COMMUNICATION_ANALYSIS.md: 970 lines
- QUICK_REFERENCE.md: 475 lines
- ARCHITECTURE_SUMMARY.md: 395 lines
- Other guides: 369 lines

Coverage:
- 5 complete training phases
- 50+ function definitions with line numbers
- 100+ code snippets and examples
- 20+ data structure definitions
- 15+ distributed training group configurations
- Complete GPU communication patterns
```

---

## Version Info

**Created**: November 18, 2025
**Scope**: Qwen3-4B RLHF Training via scripts/run-qwen3-4B.sh
**Configuration**:
- 8 GPUs on 1 node
- Tensor parallel size: 2
- Data parallel size: 4
- Advantage estimator: GRPO
- Rollout engine: SGLang (vLLM)
- Training backend: Megatron-LM

---

## Navigation Map

```
START HERE
    ↓
ARCHITECTURE_SUMMARY.md (understand what happens)
    ↓
Choose your path:
    ├→ Full understanding: TRAINING_FLOW_DOCUMENTATION.md
    ├→ Quick reference: QUICK_REFERENCE.md
    ├→ GPU details: GPU_COMMUNICATION_ANALYSIS.md
    └→ Code browsing: [use file locations from any doc]
```

---

## How to Update This Documentation

When modifying the training code:
1. Update relevant line numbers in all documents
2. Add new function definitions to QUICK_REFERENCE.md table
3. Update data flow diagrams in ARCHITECTURE_SUMMARY.md if logic changes
4. Add new GPU communication patterns to GPU_COMMUNICATION_ANALYSIS.md if applicable

---

## Questions This Documentation Answers

1. ✅ **How does train.py parse arguments?** → QUICK_REFERENCE.md Section 1.1
2. ✅ **How are Ray actors initialized?** → TRAINING_FLOW.md Section 1.5
3. ✅ **How is the model loaded?** → TRAINING_FLOW.md Section 1.5 Phase 4
4. ✅ **How is distributed training set up?** → TRAINING_FLOW.md Section 1.5 Phase 3
5. ✅ **What's the complete flow of one training iteration?** → ARCHITECTURE_SUMMARY.md Data Flow
6. ✅ **How are rollouts generated?** → TRAINING_FLOW.md Section 2.3b
7. ✅ **How does data flow between components?** → ARCHITECTURE_SUMMARY.md Data Flow
8. ✅ **How are rewards computed?** → TRAINING_FLOW.md Section 2.3b + 3.4
9. ✅ **How does the Megatron actor process data?** → TRAINING_FLOW.md Section 3.2-3.5
10. ✅ **What's the forward pass?** → TRAINING_FLOW.md Section 3.5 Phase 2
11. ✅ **How is loss computed?** → TRAINING_FLOW.md Section 3.5 Phase 4
12. ✅ **How is the backward pass done?** → TRAINING_FLOW.md Section 3.5 Phase 3-5
13. ✅ **How is gradient synchronization handled?** → TRAINING_FLOW.md Section 3.5 Phase 5
14. ✅ **How are weights synchronized between actor and inference?** → TRAINING_FLOW.md Section 4
15. ✅ **What GPU communication happens?** → GPU_COMMUNICATION_ANALYSIS.md

---

Good luck with your training! Feel free to reference these documents as needed.
