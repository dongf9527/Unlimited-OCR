# 用户环境依赖版本适配评估

针对使用者提供的实测环境（CNBEJFSOP03 `~/sourcecode/ppt-master`，pip list 摘录，2026-10-27）逐项对照本文档库的版本基线，回答"现有版本能否直接适配 PDF-OCR 管线"。基线来源：`docs/prd/pdf-ocr-prd.md` §5.1 与 `docs/prd/deployment.md` §2/§3；自动化复检工具见 `scripts/check_env.py`（本次评估的全部判定逻辑已固化在该脚本中，可在目标机器一键复核）。

## 1. 逐项对照结论

| 组件 | 文档基线 | 实测版本 | 判定 | 说明 |
| --- | --- | --- | --- | --- |
| torch | 2.13.0 + cu128 | 2.11.0 | 路线 A 可试 ／ 路线 B 不可用 | 模型代码仅用标准算子，路线 A 大概率可跑但偏离验证基线；路线 B 的 SGLang wheel 按 2.13.0 编译，硬性要求一致 |
| torchvision | 与 torch 同源 | 0.26.0 | ✓（配对正确） | torch 2.11 ↔ torchvision 0.26 属官方配对 |
| torchaudio | 管线不使用 | 2.11.0 | 无影响 | |
| transformers | 4.52.1（必须） | 5.19.0 | **✗ 路线 A 阻断** | v5 大版本重构/移除了 trust_remote_code 模型源码依赖的内部 API，详见 §2.1 |
| tokenizers | 0.21.4 | 0.23.2 | 与 transformers 5.19 自洽；不可单独保留 | 0.23.x 恰为 transformers 5.x 的配套；回退 transformers 时必须成组回退 0.21.4 |
| pillow | ≥ 10.0 | 11.0.0 | ✓ | |
| opencv-python-headless | ≥ 4.9 | 5.0.0.93 | ✓（附带条件） | 满足下限但跨 5.x 大版本；管线仅用基础 IO/裁剪 API，建议随冒烟一并确认 |
| python-docx | ≥ 1.1 | 1.2.0 | ✓ | 导出层 |
| pymupdf | 1.27.2.2 | 未在摘录中 | 待检 | PDF 渲染必需，用 check_env.py 确认 |
| einops / addict / easydict | 0.8.2 / 2.4.0 / 1.13 | 未在摘录中 | 待检 | 模型源码确证 import，缺失即无法加载 |
| huggingface-hub | 0.36.0 | 未在摘录中 | 待检 | 权重下载链路 |
| reportlab | ≥ 4.0 | 未在摘录中 | 待检 | 文本型 PDF 导出层 |
| sglang + fastokens | wheel 0.5.20.dev10 / 0.3.2 | 未在摘录中 | 待检 | 仅路线 B 需要 |
| pytorch-crf / torch_c_dlpack_ext / sentence-transformers | 管线不使用 | 0.7.2 / 0.1.5 / 6.0.0 | 无影响 | 但它们的存在表明这是共享科研环境，见 §3 建议 1 |

## 2. 关键判定依据

### 2.1 transformers 5.19.0：路线 A 的硬阻断

判定不是基于"版本号不同"的经验主义，而是基于对仓库内 `hf-model-code/` 建模源码的静态检查：trust_remote_code 机制会把模型目录下的 `modeling_unlimitedocr.py` 及其依赖链（`_sparse_base_impl.py`、`modeling_deepseekv2.py`、`dash_attention.py` 等）在运行时 import 进 transformers 的插件体系，这些文件直接引用了 4.x 的内部 API。具体符号清单如下，全部取自源码 import 语句：

- `transformers.pytorch_utils`：`ALL_LAYERNORM_LAYERS`、`is_torch_greater_or_equal_than_1_13`（版本哨兵常量，v5 清理 torch<2 支持后此类常量属被裁剪对象）；
- `transformers.models.llama.modeling_llama`：`LlamaAttention`、`repeat_kv`、`apply_rotary_pos_emb`、`LlamaRotaryEmbedding`（跨模型私有符号借用，v5 对 llama 建模与注意力接口做过整体重构，函数签名与类结构均已变化）；
- `transformers.modeling_attn_mask_utils`：`_prepare_4d_causal_attention_mask`（私有函数，v5 重写了掩码处理路径）；
- `transformers.utils.import_utils`：`is_torch_fx_available`（torch.fx 追踪集成在 v5 中已移除）。

外部事实与此互相印证：transformers v5.0.0 已于 2026 年 1 月转正（官方发布说明称其为"一次大幅内部清理"），tokenization 后端同步重新设计；HuggingFace 官方仓库 issue #43611 明确记录了 v5.0.0 破坏部分远程代码模型的加载、且 4.x 全系无此问题。PRD 将 transformers 钉死在 4.52.1 并标注"必须，trust_remote_code 依赖此版行为"，针对的正是这一风险。结论：除非将整套 remote code 源码迁移到 v5 API（工作量等同一个小型适配项目，且需上游配合），路线 A 必须回退 4.52.1。

### 2.2 torch 2.11.0：路线 A 可试，路线 B 禁止

路线 A（Transformers 单机）方面，模型源码使用的都是常规算子（Linear、SDPA、einops 重排、bf16 数据类型），没有依赖 2.12/2.13 的新特性，2.11.0 大概率可以跑通；但 2.13.0 是上游验证过的基线，偏离基线意味着 CUDAGraph、自定义核等性能特性不在保证范围，因此判定为"可试但需冒烟确认"而非"可用"。附带的好消息是 torchvision 0.26.0 与 torch 2.11.0 属官方配对，无需调整。

路线 B（SGLang 服务化）方面没有讨论空间：deployment.md §3.1 归档的 SGLang wheel（0.5.20.dev10）是按 torch 2.13.0 编译的，其依赖声明对 torch 为精确 pin，SGLang 的自定义 kernel（sgl-kernel）同样按 torch ABI 编译，torch 2.11.0 会导致 wheel 依赖解析失败或运行期 ABI 错误。若走路线 B，torch 必须按文档重装 2.13.0+cu128。

### 2.3 tokenizers 0.23.2：自洽但必须随 transformers 联动

0.23.2 本身没有问题——它恰好是 transformers 5.x 系列的配套版本，这也侧面说明该环境里 transformers 5.19.0 是完整安装而非残留。问题在于联动关系：transformers 4.52.1 对 tokenizers 的声明要求是 `<0.22`，因此"只回退 transformers、保留 0.23.2"会造成 pip 依赖冲突，两者必须成组回退（transformers==4.52.1 + tokenizers==0.21.4）。同理，该环境中 sentence-transformers 6.0.0 对 transformers 有自己的版本声明，任何降级都会破坏它，这是 §3 建议 1 的直接依据。

## 3. 结论与建议

1. **不要在现有共享环境里降级混装。** CNBEJFSOP03 的环境含 sentence-transformers、pytorch-crf 等其他项目依赖，为管线单独回退 transformers/tokenizers 会破坏这些项目；且 Python 版本未经确认（PRD 要求 3.10、禁 3.12+）。
2. **按 deployment.md 建独立 venv，与现有环境并存。** `conda create -n uocr python=3.10`（或 `uv venv --python 3.10`），在 venv 内按文档装 pinned 版本。现有环境中已验证可用的 pillow 11.0.0 / python-docx 1.2.0 / opencv 5.0.0.93 可直接复用同版本号安装。
3. **若坚持复用 torch 2.11.0 走路线 A**：仅可在不装 SGLang 的纯路线 A venv 中尝试，先跑 deployment.md §2 的单页冒烟，通过后再批量；路线 B 无此选项。
4. **部署前用自检脚本复核**：`python scripts/check_env.py --route A --model-dir <权重目录>`，全绿（无 FAIL）再进冒烟测试；CI 可用 `--json` 模式收集结果。

## 4. 结论采纳状态（2026-10-28）

本文档 §2/§3 的判定已全部采纳进 PRD v1.1（`pdf-ocr-prd.md` §5.1 定稿版本表）与部署文档 v1.1（`deployment.md`）：torch 2.11.0 成为路线 A 定稿基线（torchvision 0.26.0 配对）；transformers 4.52.1 + tokenizers 0.21.4 维持强制 pin 并在部署文档中标注禁升 5.x；pillow 11.0.0 / opencv-python-headless 5.0.0.93 / python-docx 1.2.0 按实测确认版本定版；SGLang 路线 B 因与定稿 torch 栈冲突移出本期，随 v1.0 文档归档至 `docs/archive/`；环境管理定为 conda 独立环境。`scripts/check_env.py` 已同步按 v1.1 基线校验（--route B 时自动切换归档基线 2.13.0）。
