# hf-model-code — 上游 HuggingFace 建模源码

> 来源：`huggingface.co/baidu/Unlimited-OCR-2`（MIT 许可）｜ 同步日期：2026-10-27 ｜ 修订：`main`（仓库创建于 2026-10-26，最后修改 2026-10-27T12:21Z）
> 同步脚本：`scripts/sync_hf_code.sh`（含 LFS 指针检测、py 语法校验、JSON 合法性校验，可重复执行增量更新）

本目录存放 Unlimited OCR 2 的**模型建模源代码与配置**（不含权重文件）。这些文件在上游以 trust_remote_code 模式随权重分发，加载 `baidu/Unlimited-OCR-2` 时由 transformers 自动下载执行；此处独立存放以便阅读、审计与二开。许可证为 MIT。

## 文件清单

### 建模源码（10 个 .py）

| 文件 | 大小 | 作用 |
| --- | --- | --- |
| `modeling_unlimitedocr.py` | 56 KB | 主模型建模：`UnlimitedOCRForCausalLM`、`infer`/`infer_multi` 推理入口、动态切图（gundam 模式） |
| `_sparse_base_impl.py` | 222 KB | **SR-SWA 稀疏参考滑动窗口注意力**核心实现（2.0 关键创新，base 模式） |
| `dash_attention.py` | 71 KB | DASH 注意力实现 |
| `sparse_gather_triton.py` | 24 KB | Triton 稀疏 gather 算子 |
| `sparse_gather_e2e_triton.py` | 18 KB | Triton 稀疏 gather 算子（端到端版） |
| `modeling_deepseekv2.py` | 26 KB | LLM 主干（DeepSeek-V2 架构，继承 DeepSeek-OCR 路线） |
| `configuration_deepseek_v2.py` | 11 KB | 模型配置类 |
| `deepencoder.py` | 38 KB | 视觉编码器：CLIP-ViT + MlpProjector（12 种 projector 变体） |
| `packing_patch.py` | 8 KB | 序列打包补丁 |
| `conversation.py` | 9 KB | 对话模板处理 |

### 模型配置（5 个 JSON）

| 文件 | 说明 |
| --- | --- |
| `config.json` | 模型结构配置（`auto_map` 指向上方 .py，即 trust_remote_code 入口） |
| `generation_config.json` | 生成参数 |
| `tokenizer.json` / `tokenizer_config.json` / `special_tokens_map.json` | 分词器（含 `<PAGE>` 等特殊 token，page_token_id=128827） |

## 与本地权重配合使用

```python
# 权重需另行从 HF 下载（约 6.6GB safetensors，本仓库未收录）
from transformers import AutoModel, AutoTokenizer
tok = AutoTokenizer.from_pretrained("hf-model-code", trust_remote_code=True)          # 本目录可当 tokenizer 路径
model = AutoModel.from_pretrained("<权重本地路径>", trust_remote_code=True)            # 权重目录需含同名 .py 或改 auto_map
```

二开提示：直接修改本目录 .py 后，将权重文件放入本目录即可作为一个完整的本地模型仓库加载，无需重新发版；训练微调走 ms-swift（已官方支持该模型）。
