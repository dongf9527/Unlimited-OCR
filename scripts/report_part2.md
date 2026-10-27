## 4. 代码仓库走读与输出格式解析

### 4.1 仓库结构总览

仓库极其克制，核心代码只有两个 Python 文件，其余是文档与部署资产：

```
Unlimited-OCR/
├── README.md                          # v2 主文档（部署/推理/评测全流程）
├── LICENSE                            # MIT 协议
├── CONTRIBUTING.md                    # 贡献指南
├── Unlimited-OCR-2.pdf                # v2 技术报告
├── infer.py                           # 评测批量客户端（204 行）
├── utils/
│   ├── README.md                      # 输出格式与后处理规则说明
│   └── process_unlimitedOCR2.py       # 原始输出 → Markdown 转换器（380 行）
├── wheel/
│   └── sglang-0.5.20.dev10+g8d62f5fb9c-py3-none-any.whl  # 定制 SGLang 轮子（LFS）
└── assets/                            # 架构图、榜单图、演示 GIF（LFS）
```

注意两点：模型权重并不在仓库中（需从 HuggingFace/ModelScope 下载 baidu/Unlimited-OCR-2）；assets 与 wheel 使用 Git LFS 存储，浅克隆后需要 `git lfs pull` 才能拿到真实文件。模型的前向逻辑、SR-SWA 实现与推理接口全部通过 `trust_remote_code=True` 从模型仓库的代码加载——这是理解本项目工程形态的关键：**GitHub 仓库是"部署与评测工具链"，HuggingFace 模型仓库才是"模型本体代码"**。

### 4.2 infer.py：批量解析客户端的核心逻辑

infer.py 是官方跑 MPDocBench / MarathonDocBench 的批量客户端，浓缩了几个非常实用的工程模式，值得逐段拆解：

```python
# 基准对应的 max_model_len：MPDocBench 用 128K，MarathonDocBench 用 768K
BENCHMARKS = {'MPDocBench': 131072, 'MarathonDocBench': 786432}
```

- **token 预算规划（plan 函数）**：优先使用基准自带的 per-pdf token 统计表计算 `max_tokens = max_model_len - image_tokens - 700`；没有统计表时退化为 `max_tokens = 0.3 × max_model_len`（即默认给生成留 30% 预算）。若可用空间装不下，多页文档会从 gundam 回退到 base 模式重算——这套"预算感知"思路直接可以搬到生产系统。
- **服务端 refit 重试**：请求过长被服务器拒绝时，从报错信息里解析出真实的 prompt 长度，重设 `max_tokens` 后重试一次。这是与 SGLang 这类"会告诉你真实 token 数"的服务端配合的巧妙容错。
- **防复读 n-gram 处理器**：OCR 解码天然容易在表格线、装饰线上打转，请求里强制携带 `DeepseekOCRNoRepeatNGramLogitProcessor`（ngram_size=35），单页窗口 128、多页窗口 1024。这个处理器由 SGLang 的 custom_logit_processor 机制在解码侧逐 token 施加惩罚。
- **并发批量**：ThreadPoolExecutor + 单文档单请求的语义，一个文档失败不影响整体，进度实时打印 JSON 统计（页数、模式、finish_reason、耗时、PAGE 标记数）。

### 4.3 输出格式深度解析：不是 Markdown，是结构化版面块

一个极易被忽略、但对二次开发至关重要的事实：**模型原生输出不是 Markdown，而是带坐标的版面块序列**。原始输出形如：

```
<PAGE><BLOCK>title<LEVEL1>[x1,y1,x2,y2]<FORK>第一章 总则
<BLOCK>text[x1,y1,x2,y2]<FORK>为了规范……
<BLOCK>table[x1,y1,x2,y2]<FORK><table><tr><td>…</td></tr></table>
<PAGE><BLOCK>append_table[x1,y1,x2,y2]<FORK><table>…</table>
```

格式要点：

- **`<PAGE>`**：页分隔符，输出中每页一个，顺序与输入页序一致（`page_token_id=128827`，可用 custom_params 控制）；
- **`<BLOCK>label`**：块类型标签，共约 18 种——text、title（带 `<LEVELn>` 标题层级）、header、footer、page_number、image/figure/chart/seal（图/章，输出裁剪坐标）、table、image_caption、image_footnote、page_footnote、equation、ref_text、aside_text、algorithm，以及跨页前缀 `cross_*` / `append_*`；
- **`[x1,y1,x2,y2]`**：归一化到 0-1000 的边界框坐标——意味着每一个输出块都能映射回原图位置，这是做"识别结果原文溯源"、版面还原高亮的基础能力；
- **`<FORK>`**：块头与块内容的分隔符；
- **`<MERGE>`**：同页多栏合并标记；`cross_table`/`append_table` 则标记跨页表格的断片。

官方后处理脚本 process_unlimitedOCR2.py（纯标准库实现，380 行）演示了完整的"原始输出 → Markdown"流水线：按 `<PAGE>` 切分并保留空页对齐页号；`title<LEVELn>` 转 n 级标题并把标题树强制单根化；图表块转 `![](pageN_x1_y1_x2_y2.jpg)` 引用；表格保持 HTML 并对跨页表格做 **union-find 分组合并 + 重复表头去重**；清洗游离 token 与畸形 colspan/rowspan。**这段代码是二次开发最有复用价值的资产之一**——业务系统可以直接 `from process_unlimitedOCR2 import convert` 复用，或参照它实现 JSON/Docx 等自定义导出。

### 4.4 架构启示小结

把 4.2 与 4.3 合起来看，这个项目的工程哲学非常清晰：模型只负责输出"带坐标的结构化事实"，一切面向人类的格式（Markdown/HTML/JSON）都放在可替换的后处理层。这种解耦让二次开发者可以在不动模型的情况下，自由定制下游格式、做细粒度的版面分析、甚至构建"原文坐标级"的审计系统。

---

## 5. 技术栈全景清单

| 层次 | 组件 | 版本/说明 | 作用 |
| --- | --- | --- | --- |
| 语言/运行时 | Python | 3.10（v2 推荐）/ 3.12（v1 已验证） | 全栈实现 |
| 深度学习框架 | PyTorch | v2 测试版 torch==2.13.0（v1 为 2.10.0 + CUDA 12.9） | 模型推理与训练底座 |
| 模型加载 | Transformers | v2 测试 4.52.1（v1 为 4.57.1），trust_remote_code | AutoModel/AutoTokenizer 加载自定义模型代码 |
| 权重格式 | Safetensors / BF16 | 约 6.7GB | 模型权重（HuggingFace 下载） |
| 推理服务（推荐） | SGLang（**定制 wheel**） | sglang-0.5.20.dev10（仓库 wheel/ 目录提供） | 生产级服务：连续批处理、CUDA Graph、页窗口调度、OpenAI 兼容 API |
| 推理服务（备选） | vLLM | 社区支持（2026-06-28 起支持 v1） | 已有 vLLM 基础设施的团队可复用 |
| 快速验证 | Transformers 原生 | model.infer / model.infer_multi | 单机测试、小批量任务 |
| 分词加速 | fastokens | 0.3.2，--tokenizer-backend fastokens | SGLang 服务端分词后端 |
| 文档预处理 | PyMuPDF (fitz) | 1.27.2.2 | PDF → 300dpi 页图 |
| 轻量依赖 | einops / addict / easydict / pillow | 0.8.2 / 2.4.0 / 1.13 / 最新 | 模型代码依赖 |
| 环境管理 | uv | --python 3.10 | 官方推荐的虚拟环境方案 |
| 训练/微调 | ms-swift | 社区支持（2026-07-21 起） | 官方认可的微调框架，支持 LoRA/全参 |
| 容器化 | Docker Model Runner | 模型卡列出的部署方式之一 | 容器化部署 |
| 评测生态 | MPDocBench / MarathonDocBench / OmniDocBench | Marathon 为百度自建（100 份 103-300 页文档，20.5 万标注版面元素） | 精度评测 |
| 托管生态 | HuggingFace / ModelScope / 百度智能云 / 百度AI开放平台 / HF Spaces Demo | 下载量 1000 万+ | 分发与在线体验 |

**技术栈特征总结**：这是一个"Python 单语言 + HuggingFace 生态原生 + 服务化靠 SGLang 深度定制"的项目。最大的工程门槛与最深的护城河都在那颗定制 SGLang wheel 上——页窗口调度、边界页 KV 复用、SAM 注意力偏置优化等能力都是改进去，上游 SGLang 暂不包含。

---

## 6. 核心功能盘点

1. **单页高精度解析（gundam 模式）**：动态切片编码，适合论文、票据、合同、密集表格等复杂版面，可还原标题层级、表格结构、公式（LaTeX）、图片位置、印章、脚注与阅读顺序。单页是模型训练的主场景（1000 万单页样本），精度最高。
2. **多页一次请求解析（infer_multi / base 模式）**：整份文档的页图按序传入，prompt 自动构造为 `"<PAGE><image>" × N + "Multi page parsing."`，模型在单次推理中利用跨页上下文完成全书转录，输出按页对齐。跨页断句、跨页表格能被正确缝合（v2 相对 v1 的决定性优势）。
3. **PDF 直接解析**：官方示例提供 PyMuPDF 300dpi 转图函数 `pdf_to_images()`，扫描件与原生 PDF 统一走"页图序列"入口。
4. **结构化输出与坐标溯源**：见 4.3 节——块类型 + 0-1000 归一化 bbox + 阅读顺序，可支撑原文高亮、版面还原、审计比对等高阶需求。
5. **跨页表格自动合并**：cross_table / append_table 断片自动拼接、重复表头自动去重——财报、总账类文档的刚需能力（媒体实测 300 页 3000 笔流水的账本，925 万总额对账精确到分）。
6. **OpenAI 兼容 API + 流式输出**：SGLang 服务暴露 `/v1/chat/completions`，支持 stream，可直接接入现有 OpenAI SDK 生态；`images_config.crop_mode` 控制编码模式，custom_logit_processor 控制防复读。
7. **极端长度稳定性**：官方与媒体实测 3623 页（哈利·波特全系列）至 5026 页（四大名著合订本）一次请求解析到底，末页内容仍准确（实测含防"背书脑补"的伪字校验，模型按纸面原样转录而非记忆纠正）。

---

## 7. 本地部署实战指南（NVIDIA GPU）

### 7.1 硬件要求分析

官方在 **NVIDIA A800-80GB** 上完成了 Transformers 与 SGLang 两条路线的测试。但拆开看各项显存构成，消费级显卡并非没有机会：

| 组件 | 显存/资源估算 | 说明 |
| --- | --- | --- |
| 模型权重（BF16） | 约 6.7GB | 3.34B 参数 BF16 静态占用 |
| 单页/少页推理 KV | 很小（每页 273 image token） | 数十页文档 KV 仍在 GB 级 |
| 长文档解码 KV | SR-SWA + 窗口化后单序列约 1.77GB（百页级实测） | 无限解码机制的直接收益 |
| SGLang 服务开销 | CUDA Graph 捕获 + 请求槽 + 预分配 KV 池 | mem-fraction-static 0.7 起步 |
| 系统内存 | 建议 ≥32GB | 页图编码、base64 传输、批量任务 |
| 磁盘 | ≥20GB 空闲 | 权重 6.7GB + 环境 + 页图临时目录 |

**给 NVIDIA GPU 用户的务实建议**：
- **24GB 卡（RTX 4090/3090）**：跑 Transformers 路线的单页与中小批量（≤10 页）推理完全可行（权重 6.7GB + 激活 + KV 余量充足）；SGLang 服务需压缩配置（降低 max-running-requests、减小页窗口、mem-fraction 降到 0.6 左右试验）。
- **48GB 卡（A6000/L40S）**：SGLang 服务化舒适区，官方参数基本可直接套用。
- **80GB 卡（A800/H800/A100）**：官方完整复现环境，64 并发、长文档全量参数无压力。
- 显存不是唯一瓶颈：**长文档总耗时随页数线性增长**（每页都要视觉编码），3600 页实测耗时约 1.5 小时即 1.5 秒/页，规划批量任务时请按此量级预估。

### 7.2 路线 A：Transformers 快速上手（5 分钟验证）

```bash
# Python 3.10 环境
pip install torch==2.13.0 torchvision pillow transformers==4.52.1 \
    huggingface-hub==0.36.0 tokenizers==0.21.4 \
    addict==2.4.0 einops==0.8.2 easydict==1.13
```

```python
import torch
from transformers import AutoModel, AutoTokenizer

model_name = "baidu/Unlimited-OCR-2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True,
                                  use_safetensors=True,
                                  torch_dtype=torch.bfloat16).eval().cuda()

# 单页：gundam 动态切片
text = model.infer(tokenizer, image_file="page_1.jpg",
                   output_path="output/single", save_results=True)

# 多页：一次请求整本（base 模式，跨页上下文）
pages = [f"your_doc/page_{i}.jpg" for i in range(1, 8)]
text, num_tokens = model.infer_multi(tokenizer, image_files=pages,
                                     output_path="output/multi", save_results=True)
```

### 7.3 路线 B：SGLang 生产级服务（官方推荐）

```bash
# uv 管理虚拟环境
uv venv --python 3.10
source .venv/bin/activate
uv pip install wheel/sglang-0.5.20.dev10+g8d62f5fb9c-py3-none-any.whl  # 必须用定制 wheel
uv pip install pymupdf==1.27.2.2
uv pip install fastokens==0.3.2
```

关键环境变量（官方逐条注明了存在理由，部署前务必过一遍）：

```bash
export SGLANG_UOCR_EANABLE_DOC_LIMIT=1        # 1=启用页窗口调度；0=整本文档单窗口
export SGLANG_UOCR_DOC_GUNDAM_PAGE_WINDOW=10  # gundam 模式每窗口页数
export SGLANG_UOCR_DOC_BASE_PAGE_WINDOW=10    # base 模式每窗口页数
export SGLANG_UOCR_MAX_PAGE_TOKENS=20000      # 每页输出 token 上限（必须>0）
export SGLANG_OPT_UOCR_SAM_REL_POS_BIAS_MB=64 # SAM 注意力偏置分块，默认512会一次性申请约462MiB导致OOM
export SGLANG_UOCR_SHM_WRAP_MODEL_DATA=1      # 大crop张量走/dev/shm而非pickle
export SGLANG_UOCR_SHM_WRAP_MIN_MB=1
export SGLANG_VLM_CACHE_SIZE_MB=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 降低大块连续分配的碎片失败
```

启动服务（与显存强相关的参数已注释）：

```bash
python -m sglang.launch_server \
  --model-path baidu/Unlimited-OCR-2 \
  --served-model-name Unlimited-OCR-2 \
  --chat-template unlimited-ocr \
  --tokenizer-backend fastokens --tokenizer-worker-num 8 \
  --mem-fraction-static 0.7 \            # 0.75 在 32+ 并发时余量不足，0.7 稳定
  --max-running-requests 64 \
  --max-prefill-tokens 2048 \            # 临时张量 [tokens, ctx/32]，减半可省一半峰值
  --enable-custom-logit-processor \
  --enable-streaming-session --allow-auto-truncate --disable-radix-cache \
  --mm-preprocess-cache-size-mb 0 \
  --cuda-graph-backend-decode full --cuda-graph-max-bs-decode 64 \
  --cuda-graph-backend-prefill full \
  --cuda-graph-max-bs-prefill 16384 \
  --cuda-graph-bs-prefill 1024 2048 \    # 最大桶必须等于 max-prefill-tokens
  --cuda-graph-config '{"prefill": {"backend": "full", "full_prefill_max_req": 64}}' \  # 默认8会使16/32并发回退eager
  --host 0.0.0.0 --port 10000
```

### 7.4 调用端：OpenAI 兼容 API

```python
payload = {
    "model": "Unlimited-OCR-2",
    "messages": [{"role": "user", "content": [图片content块...] +
                  [{"type": "text", "text": "<PAGE><image>" * n + "Multi page parsing."}]}],
    "max_tokens": 8000 * n_pages,          # 全文档总输出预算，必须为正
    "temperature": 0,
    "skip_special_tokens": False,          # 保留<PAGE>等标记，后处理依赖它们
    "images_config": {"crop_mode": True},  # True=gundam 切片；False=base 全局视图
    "custom_logit_processor": NGRAM_PROCESSOR,
    "custom_params": {"ngram_size": 35, "window_size": 1024,
                      "page_token_id": 128827, "eos_token_id": 1,
                      "max_page_tokens": 20000},
    "stream": True,
}
# POST http://127.0.0.1:10000/v1/chat/completions
```

### 7.5 常见坑清单（部署前必读）

1. **462MiB OOM**：SAM 相对位置偏置默认一次申请约 462MiB，显存紧张时必设 `SGLANG_OPT_UOCR_SAM_REL_POS_BIAS_MB=64`（官方确认数值等价、峰值更低）。
2. **cuda-graph 桶不匹配**：`--cuda-graph-bs-prefill` 最大桶若小于 `--max-prefill-tokens`，超限批次静默回退 eager 模式，吞吐骤降且无报错。
3. **full_prefill_max_req 默认值陷阱**：默认 8，16/32 并发时会回退 eager，必须显式配到实际并发档位。
4. **页窗口与跨页质量**：窗口越小峰值显存越低，但窗口边界的跨页断句/续表质量会受损——精度敏感场景优先调大窗口而不是调小。
5. **max_tokens 语义**：是"全文档总输出预算"而非单页，多页文档必须按 `tokens_per_page × n_pages` 给足。
6. **skip_special_tokens 必须 False**：否则 `<PAGE>` 标记被吞掉，页对齐与后处理全部失效。
7. **trust_remote_code=True**：会执行模型仓库中的自定义 Python 代码，生产环境建议固定 revision 并审计代码。
8. **wheel 版本锁定**：定制 SGLang 与上游独立演进，升级前先看仓库 README 是否有配套说明。

---
