# Multi-Latent Attention 分析及其在 TP/SP 下与 GQA 的区别

## MLA 实现概述

Multi-Latent Attention ("MLA") 是 Deepseek 团队提出的一种注意力机制，其核心实现位于 `megatron/core/transformer/multi_latent_attention.py`。MLA 通过对 Query/Key/Value 进行低秩下采样（``linear_q_down_proj``、``linear_kv_down_proj``），再在计算前执行上采样并施加旋转位置编码，从而在保持表达能力的同时减少 KV 缓存与计算量。

MLA 支持 Tensor Parallel (TP) 与 Sequence Parallel (SP) 环境。在函数 ``get_query_key_value_tensors`` 中，会根据是否启用 TP/SP 对张量进行 ``gather`` 与 ``scatter``，确保分布式场景下形状与行为与单卡一致。其配置通过 `MLATransformerConfig` 指定，相关文档位于 `docs/source/api-guide/multi_latent_attention.rst`。

### Codemap

| 文件路径 | 说明 |
| --- | --- |
| `megatron/core/transformer/multi_latent_attention.py` | MLA 核心实现，包含下采样、上采样与 RoPE 处理 |
| `megatron/core/transformer/transformer_config.py` | `MLATransformerConfig` 定义 MLA 各项超参 |
| `megatron/core/models/common/embeddings/rope_utils.py` | RoPE 工具，支持 `multi_latent_attention` 参数 |
| `megatron/core/models/gpt/gpt_layer_specs.py` | GPT 层规范，配置 MLA 所需模块 |
| `megatron/training/arguments.py` | 训练脚本的命令行参数，如 `--multi-latent-attention` |
| `docs/source/api-guide/multi_latent_attention.rst` | MLA 简要使用文档 |

## MLA 在 TP/SP 下的处理

在 `get_query_key_value_tensors` 中，若下采样层 (`ColumnParallelLinear`) 的输出被分片，需要显式收集并在必要时重新分散。示例代码如下：

```python
if q_compressed.size(-1) != self.config.q_lora_rank:
    q_compressed = gather_from_tensor_model_parallel_region(q_compressed)
    if self.config.sequence_parallel:
        q_compressed = scatter_to_sequence_parallel_region(q_compressed)
```
【F:megatron/core/transformer/multi_latent_attention.py†L388-L404】

Key/Value 的处理逻辑与此类似：

```python
kv_combined, _ = self.linear_kv_down_proj(hidden_states)
if kv_combined.size(-1) != self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim:
    kv_combined = gather_from_tensor_model_parallel_region(kv_combined)
    kv_compressed, k_pos_emb = torch.split(
        kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
    )
    if self.config.sequence_parallel:
        kv_compressed = scatter_to_sequence_parallel_region(kv_compressed)
```
【F:megatron/core/transformer/multi_latent_attention.py†L410-L424】

这种显式的 gather/scatter 过程保证了在下采样-上采样阶段各 rank 均能拿到完整的张量，从而在 TP/SP 环境下保持计算一致。

## GQA 的 TP/SP 处理

Grouped Query Attention (GQA) 通过减少 KV 头数来降低 KV 缓存和通信量，其实现沿用常规 `Attention`。在 `Attention.get_query_key_value_tensors` 中，会直接从 `linear_qkv` 得到合并后的张量并按照分组维度拆分：

```python
mixed_qkv, _ = self.linear_qkv(hidden_states)
new_tensor_shape = mixed_qkv.size()[:-1] + (
    self.num_query_groups_per_partition,
    (
        (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
        * self.hidden_size_per_attention_head
    ),
)
mixed_qkv = mixed_qkv.view(*new_tensor_shape)
(query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)
```
【F:megatron/core/transformer/attention.py†L899-L935】

由于 `ColumnParallelLinear` 已在内部完成张量的并行分布，GQA 不需要像 MLA 那样额外执行 gather/scatter 操作。

## MLA 与 GQA 在分布式操作上的差异

1. **下采样与上采样流程**：MLA 需在 TP 环境中先将压缩后的 Q/K/V 聚合，再进行上采样与位置编码；GQA 则直接通过分组化的 QKV 线性层得到所需张量，无额外聚合步骤。
2. **对 SP 的处理**：MLA 在恢复下采样维度后，如开启 Sequence Parallel，需要再次将张量按照序列划分；GQA 的 QKV 张量在分组拆分后即可直接用于计算。
3. **KV Cache 维度**：MLA 使用低秩投影减小缓存维度，并在 TPU/SP 中通过 gather/scatter 保持维度一致；GQA 依靠减少 KV 头数来压缩缓存，不涉及额外的张量收集。

综上，MLA 在 TP/SP 下需要显式的数据收集与分散，而 GQA 由于其设计上的简化，分布式操作更为直接。
