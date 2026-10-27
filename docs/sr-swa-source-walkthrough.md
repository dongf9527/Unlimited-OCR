# SR-SWA 源码导读

> 基于 `hf-model-code/` 实际读码（2026-10-27）｜ 核心文件：`_sparse_base_impl.py`（222KB，全部源码中最大）

## 一、SR-SWA 是什么

SR-SWA（Sparse Reference Sliding Window Attention，稀疏参考滑动窗口注意力）是 Unlimited OCR 2（2026-10-27 发布）的核心架构创新，解决超长文档解析中注意力二次方膨胀问题：解码每一步只维护**固定大小的"工作参考集"**（约 512 token 量级），使单步解码延迟与显存占用几乎不随文档页数增长。官方基准：百页文档场景单步解码速度较 Full Attention 提升 7 倍、较上一代 R-SWA 提升 5.6 倍。

## 二、源码载体与血统

文件头声明基于 DeepSeek-AI + HuggingFace 实现（Apache 2.0）改造，"PyTorch DeepSeek model compatible with DeepSeekV2/V3"。三个注意力类分工：

| 类（行号） | 角色 |
| --- | --- |
| `DeepseekV2Attention`（843） | 标准 MLA 注意力（1D RoPE），作为兼容基线；收到 page2d 配置时打印警告不生效 |
| `DeepseekV2FlashAttention2`（1082） | FA2 变体 |
| **`SlidingWindowLlamaAttention`（1366）** | **SR-SWA 真正的运行载体**，继承 LlamaAttention 深度改造 |

## 三、`SlidingWindowLlamaAttention` 内的四个机制

1. **块稀疏索引器 `DeepseekV2Indexer`（2277）**：DSA（DeepSeek Sparse Attention）血统的 lightning indexer，按块（`block_size`）为每个 query 挑选最相关的 KV 块（top-k），配 `DeepseekV2IndexerRotaryEmbedding`（1935）与 `DeepseekV2IndexerRotary2D`（1984）两套索引位置编码。

2. **Anchor/Reuse 分层**：`sparse_attention_freq` 指定哪些层做稀疏；`sparse_inconsistency_freq=1` 的层为 Anchor（自己算 block_indices），`=0` 的层为 Reuse（复用最近 Anchor 层的索引结果）。省算力的同时保证 warmup（全 1）与稀疏阶段（0/1 混合）的 state_dict key 完全一致，checkpoint 兼容。

3. **页面二维位置编码 page2d**：`sparse_main_rope_mode=page2d` 时启用 `DeepseekV2IndexerRotary2D`，参数 `sparse_rope_page_pairs=8`、`sparse_rope_page_theta=256.0`——把"第几页"作为独立维度编进 RoPE，即模型卡所称"页面门控 Page Gate 建模跨页位置关系"的代码实体。

4. **窗口与视觉桶**：`local_window` 局部滑动窗口（config.json `_ring_window=128`）；`image_token_id` + `image_bucket_constraint` 实现"页内视觉 token 桶"——页首图整块入桶；forward 中 pre/post mask 对应"page-image bucket + causal/packed text"两段掩码。

## 四、配置驱动与部署钩子

全部行为由 `config.json` 的 `sparse_attention_config` 字典 + 环境变量驱动，无独立新模块文件。SGLang 侧钩子（`SGLANG_UOCR_DOC_GUNDAM_PAGE_WINDOW` 等页窗调度参数）在 `_sparse_base_impl.py` 与 `dash_attention.py` 中消费；`TIME_DIAG=1` 可开启分区耗时诊断。配套 Triton 算子在 `sparse_gather_triton.py`（分步）与 `sparse_gather_e2e_triton.py`（端到端）。

## 五、二开切入点

- 调整 `_ring_window` / `block_size` / top-k 比例：精度-显存权衡的最低成本实验位；
- `sparse_attention_freq` / `sparse_inconsistency_freq`：逐层稀疏密度实验（注意 Anchor/Reuse 依赖校验会显式报错，见 1416 行 `SparseReuseConfigError`）；
- page2d 参数（page_pairs/page_theta）：跨页依赖建模强度；
- 改造前先用 `TIME_DIAG=1` 定位瓶颈区域再动手。
