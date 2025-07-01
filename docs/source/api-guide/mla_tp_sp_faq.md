# MLA 在 TP/SP 下的关键问题整理

本文件总结了与 Multi-Latent Attention (MLA) 实现及其在 Tensor Parallel (TP) 和 Sequence Parallel (SP) 场景下表现相关的常见疑问，源自我们的内部讨论。

## 1. `q_lowrank` 的位置与作用

- `q_lowrank` 即 `linear_q_down_proj`，位于 Transformer 层开头，用于将隐藏状态从 `hidden_size` 压缩到 `q_lora_rank` 维度。
- 它并不是在其他并行层之后再接入，而是直接替代标准注意力中 `linear_q_proj` 的作用。
- 若该层实现为 `ColumnParallelLinear`，其输出会在 TP 维度上分片。

## 2. TP/SP 下的 gather/scatter

下采样得到的张量在 TP 环境下需要显式聚合，再在需要时重新按序列拆分。代码示例如下：

```python
if q_compressed.size(-1) != self.config.q_lora_rank:
    q_compressed = gather_from_tensor_model_parallel_region(q_compressed)
    if self.config.sequence_parallel:
        q_compressed = scatter_to_sequence_parallel_region(q_compressed)
```
【F:megatron/core/transformer/multi_latent_attention.py†L388-L404】

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

这些操作保证了下采样后的张量在各个 rank 上具有完整维度，随后才能正确进行上采样和旋转位置编码。

## 3. 为何要这样处理？

- 如果直接使用 `ColumnParallelLinear` 的输出进行上采样，会导致每个 rank 只持有部分维度，后续操作与单卡结果不一致。
- 通过在 down projection 后手动 gather，再根据是否使用 SP 将序列重新拆回，能够保持与单卡运行时完全一致的形状和语义。

## 4. 与 GQA 等传统实现的差异

GQA 通常通过减少 KV 头数来降低开销，其 `linear_qkv` 直接在高维空间生成 Q、K、V，并依靠 `ColumnParallelLinear` 自动完成并行分布。GQA 不需要手动 gather/scatter。这一点在文档中有详细对比。

【F:docs/source/api-guide/mla_vs_gqa.md†L1-L74】

MLA 则使用低秩下采样-上采样的方式，需要显式地在 TP/SP 环境下同步张量，以确保后续计算的一致性，并降低 KV 缓存的维度。

## 5. 数据 shape 流简述

以典型的 TP+SP 环境为例：

1. 输入 `hidden_states` 形状为 `[s/TP, b, h]`；
2. 经过 `linear_q_down_proj` 得到 `[s/TP, b, q_lora_rank]`，若为 `ColumnParallelLinear`，先聚合再按序列拆分；
3. 同理处理 `linear_kv_down_proj`，拆分出 `kv_compressed` 与 `k_pos_emb`；
4. 上采样 (`linear_q_up_proj` / `linear_kv_up_proj`) 并加上旋转位置编码，得到完整的 Q/K/V；
5. 随后的注意力计算与标准实现保持一致。

通过这些步骤，MLA 能够在大规模模型的分布式场景下高效运行，同时减少内存和通信成本。

