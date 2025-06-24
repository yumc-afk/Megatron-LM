# Megatron 分布式检查点加载过程问答整理

本文档整理了关于 Megatron 分布式检查点 (dist checkpoint) 加载流程的常见问答，主要基于先前的对话记录，总结其工作原理与涉及的通信操作。

## 1. 多个 rank 如何从同一分片文件按需读取？

PyTorch Distributed Checkpoint 在元数据中记录每个张量在文件中的偏移量与大小 (`TensorStorageMetadata`)。加载时 `FileSystemReader` 会打开文件并通过 `seek/readinto` 只读取所需的字节，这样各个 rank 即便共享同一个分片文件，也只会读取自己负责的区域，避免重复 I/O。

## 2. dp=1 时是否会发生跨 rank 的通信？

`FullyParallelLoadStrategyWrapper.load()` 会先判断并行组大小。如果数据并行组只有一个 rank（即 `dp=1`），将直接调用底层加载策略而不进行任何通信；每个张量分片都由该 rank 从磁盘读取，无需与其他进程交换。

## 3. LoadPlanner 与 LoadPlan 的作用

`checkpoint.load_state_dict` 结合 `LoadPlanner` 生成 `LoadPlan`，确定各 rank 读取哪些分片、需要向哪些 rank 发送数据。Megatron 使用继承自 `DefaultLoadPlanner` 的 `MCoreLoadPlanner`，在保留 PyTorch 默认逻辑的同时增加形状校验等功能。

## 4. 复用 PyTorch Distributed 的哪些特性？

在 `TorchDistLoadShardedStrategy.load()` 中，MCore 的 `ShardedTensor` 会被转换为 `TorchShardedTensor`，再调用 `checkpoint.load_state_dict`。通过这种方式，Megatron 复用了 PyTorch 在元数据规划、异步存取、跨 rank 协调等方面的实现，加载完成后再将 `TorchShardedTensor` 转回 `ShardedTensor`。

## 5. MCore 的 `ShardedTensor` 与 `TorchShardedTensor` 区别

- `ShardedTensor` 是 MCore 定义的结构，记录更多与模型相关的分片信息，如 `replica_id`、`flattened_range` 等。
- `TorchShardedTensor` 是 PyTorch 官方的分布式张量类型，能被 `torch.distributed.checkpoint` 直接处理。

Megatron 在保存或加载前会把前者转换为后者，以便使用 PyTorch 的检查点格式，加载完成后再还原。

## 6. 各 rank 之间为何需要交换已加载的张量？

当数据并行组规模大于 1 时，为了减少磁盘 I/O，通常让每个 rank 只读取部分分片，然后通过 `exchange_by_distribution` 将已加载的张量发送给需要它们的其他 rank。这一过程可以通过 `broadcast`、多次 `all_gather` 或调试用的 `all_gather_object` 等方式实现。若 `dp=1`，则不会进行这些交换。

对于 `ShardedObject` 等非张量数据，默认通过 `all_gather_object` 在各个 rank 之间互相收集。

## 7. 具体加载流程及涉及的通信

以下以 `FullyParallelLoadStrategyWrapper` 为例说明整体流程：

1. **交换元信息并确定分片分配**：使用 `torch.distributed.all_gather_object` 收集各 rank 的元数据，计算每个 rank 负责加载哪些分片。
2. **各 rank 加载本地分片**：调用底层策略（如 `TorchDistLoadShardedStrategy.load`）从存储读取分片。
3. **交换已加载的张量**：根据配置选择 `broadcast`、`all_gather` 等方式互相发送张量分片。
4. **交换对象数据**：通过 `all_gather_object` 交换 `ShardedObject` 的内容。
5. **组装完整的 state dict**：将收到的张量和对象插入回原先的 `sharded_state_dict`，返回模型权重。

## 8. `load_distributed_checkpoint` 示例

```python
from megatron.core.dist_checkpointing import serialization as dist_checkpointing

def load_distributed_checkpoint(checkpoint_path, gpt_model):
    sharded_state_dict = gpt_model.sharded_state_dict(prefix='')
    checkpoint = dist_checkpointing.load(
        sharded_state_dict=sharded_state_dict,
        checkpoint_dir=checkpoint_path)
    gpt_model.load_state_dict(checkpoint)
    return gpt_model
```

该函数首先获取模型的 `sharded_state_dict`，随后通过 `dist_checkpointing.load` 完成加载。默认策略即 `FullyParallelLoadStrategyWrapper`，按前述步骤实现从存储到通信的完整流程。

## 9. 调用链路图

下图展示了 `load_distributed_checkpoint` 的主要调用链（缩写 `FPLSW` 代表 `FullyParallelLoadStrategyWrapper`），括号内注明关键细节：

```
load_distributed_checkpoint
    |
    v
dist_checkpointing.load
    |
    v
FPLSW.load
    |-- determine_main_replica_uniform_distribution (all_gather_object 收集元信息)
    |-- base_strategy.load -> TorchDistLoadShardedStrategy.load
    |        |-- _replace_state_dict_keys_with_sharded_keys
    |        |-- mcore_to_pyt_state_dict
    |        `-- checkpoint.load_state_dict(FileSystemReader, planner)
    |-- exchange_by_distribution (broadcast / all_gather 交换张量)
    `-- exchange_loaded_objects_gather_object (all_gather_object 交换对象)
```

可以看到，除本地磁盘读取外，通信主要发生在 `all_gather_object`（收集元数据与对象）和 `broadcast`/`all_gather`（分发张量）等步骤；若数据并行组仅有一个 rank，上述通信会被全部跳过。
