# Unlimited-OCR 源代码分布调研

> 调研日期：2026-10-27 ｜ 结论先行：**Unlimited-OCR 是完整开源的（MIT 许可，含权重与全部建模源码），但源代码并不在 GitHub 仓库里**，而是以 trust_remote_code 模式随模型权重托管在 HuggingFace / ModelScope。

## 一、问题背景

`baidu/Unlimited-OCR` GitHub 仓库的文件清单仅为：README.md、LICENSE（MIT）、CONTRIBUTING.md、技术报告 PDF、`infer.py`（快速启动脚本）、`utils/process_unlimitedOCR2.py`（输出后处理）、`assets/`（图片）、`wheel/`（定制 SGLang wheel）。看起来"没有源代码"，容易误判为 PPT 开源。

## 二、真相：模型仓库一体化发布模式

这是社区沿用 DeepSeek-OCR 的发布惯例：**GitHub 仓库只做门面（文档 + 工程脚本），模型权重与建模代码作为一个整体托管在 HuggingFace**，加载时通过 `trust_remote_code=True` 自动拉取并执行仓库内的 .py 建模文件。三个发布渠道：

| 资产 | 位置 | 内容 |
| --- | --- | --- |
| 模型权重 + 建模源码（2.0，2026-10-27 发布） | `huggingface.co/baidu/Unlimited-OCR-2` | 3.34B 参数 safetensors（约 6.6GB）+ 10 个 .py 源文件，MIT |
| 模型权重 + 建模源码（1.0，2026-06-22 发布） | `modelscope.cn/models/PaddlePaddle/Unlimited-OCR`（HF 同名镜像） | 1.0 版权重，README 见 GitHub `UnlimitedOCR1.0` 分支 |
| 工程脚本 + 使用文档 | `github.com/baidu/Unlimited-OCR` | infer.py、后处理脚本、SGLang 部署文档与定制 wheel |
| 在线体验 | `huggingface.co/spaces/baidu/Unlimited-OCR` | HF Spaces Demo |

## 三、HuggingFace 仓库内的源代码清单（2.0）

经 HF API 核实（`huggingface.co/api/models/baidu/Unlimited-OCR-2`），源代码共 10 个文件：

| 文件 | 作用 |
| --- | --- |
| `modeling_unlimitedocr.py` | 主模型建模：多页推理入口 `infer` / `infer_multi` |
| `modeling_deepseekv2.py` | LLM 主干（DeepSeek-V2 架构，继承而来） |
| `configuration_deepseek_v2.py` | 模型配置类 |
| `deepencoder.py`（37.9 kB） | 视觉编码器：CLIP-ViT + MlpProjector（含 12 种 projector 变体） |
| `_sparse_base_impl.py` | SR-SWA 稀疏参考滑动窗口注意力的 base 模式实现（2.0 核心创新） |
| `dash_attention.py` | DASH 注意力实现 |
| `sparse_gather_triton.py` / `sparse_gather_e2e_triton.py` | Triton 稀疏 gather 算子（端到端版与分步版） |
| `packing_patch.py` / `conversation.py` | 序列打包补丁与对话模板 |

权重：`model-0000X-of-00002.safetensors` 两分片 + index（约 6.6GB）；配置：`config.json` 等 5 个。License 标签 **MIT**，可商用。

## 四、开源完整性评价

| 组成部分 | 是否开源 | 说明 |
| --- | --- | --- |
| 模型权重 | ✅ | MIT，HF/ModelScope 双渠道 |
| 推理/建模源码 | ✅ | trust_remote_code 随权重分发，Triton 算子均在 |
| 数据预处理/后处理 | ✅ | GitHub：PDF 转图 + `process_unlimitedOCR2.py` |
| 服务化部署 | ✅ | 定制 SGLang wheel（`wheel/`）+ vLLM 官方支持 |
| 训练/微调代码 | ⚠️ 部分 | 未发布官方训练框架，依赖 ms-swift 社区适配（README 2026-07-21 致谢条目） |
| 技术报告 | ✅ | arXiv 2606.23050（"Unlimited OCR Works"），仓库内同名 PDF |

结论：属于**推理全链路完整开源、训练依赖社区生态**的典型学术开源形态，非 PPT 开源。

## 五、对二开的影响与获取方式

```bash
# 方式一：huggingface-cli（含 trust_remote_code 源码一并下载）
hf download baidu/Unlimited-OCR-2 --local-dir ./Unlimited-OCR-2

# 方式二：ModelScope（国内网络友好）
git clone https://www.modelscope.cn/PaddlePaddle/Unlimited-OCR.git
```

要点：建模源码与权重同仓库，clone 即获得全部源代码；改动 `modeling_unlimitedocr.py` / `_sparse_base_impl.py` 等 .py 后本地 `from_pretrained(local_path, trust_remote_code=True)` 即可生效，无需重新发版；训练二开走 ms-swift（已官方支持该模型）。

## 六、参考来源

- HuggingFace 模型卡与 API：`huggingface.co/baidu/Unlimited-OCR-2`（下载量、文件清单、license 经 API 核实于 2026-10-27）
- GitHub README：`github.com/baidu/Unlimited-OCR`（发布时间线、SGLang 部署文档、ms-swift 致谢）
- 技术报告：arXiv 2606.23050
- 发布报道：百度官方通稿（GitHub 2.85 万 Star、HF 下载超 1000 万，1.0 发布次日四榜登顶）
