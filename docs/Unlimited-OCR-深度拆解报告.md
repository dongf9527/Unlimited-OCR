# 百度 Unlimited-OCR 深度拆解报告
## 架构 · 技术栈 · 功能 · 部署 · 二次开发全景指南

> **分析基准日**：2026-10-27（Unlimited OCR 2 发布当日）
> **信息来源**：GitHub 仓库代码走读（infer.py / utils/ / wheel/）、v1 与 v2 两版 README、arXiv 技术报告（2606.23050）、HuggingFace 模型卡、多家媒体实测报道
> **仓库地址**：https://github.com/baidu/Unlimited-OCR ｜ **模型地址**：https://huggingface.co/baidu/Unlimited-OCR-2

---

## 目录

1. [执行摘要与项目概览](#1-执行摘要与项目概览)
2. [项目背景与定位：OCR 迈向 3.0 时代](#2-项目背景与定位ocr-迈向-30-时代)
3. [系统架构深度拆解](#3-系统架构深度拆解)
4. [代码仓库走读与输出格式解析](#4-代码仓库走读与输出格式解析)
5. [技术栈全景清单](#5-技术栈全景清单)
6. [核心功能盘点](#6-核心功能盘点)
7. [本地部署实战指南（NVIDIA GPU）](#7-本地部署实战指南nvidia-gpu)
8. [基准成绩与详细横向对比](#8-基准成绩与详细横向对比)
9. [二次开发方向详解（四大路线）](#9-二次开发方向详解四大路线)
10. [局限性与风险提示](#10-局限性与风险提示)
11. [上手路线图与学习建议](#11-上手路线图与学习建议)

---

## 1. 执行摘要与项目概览

Unlimited-OCR 是百度于 2026 年 6 月 22 日首次开源、并于 2026 年 10 月 27 日迭代至 2.0 的端到端文档解析大模型（VLM-OCR）。它以 DeepSeek-OCR 为基线改造而来，核心使命是解决传统 OCR 与通用大模型在超长文档面前的"页数天花板"问题：一次请求、不切分章节地解析任意页数的文档，官方称之为"OCR 3.0 时代"的开端。项目采用 MIT 协议开源，代码与权重同步发布于 GitHub 与 HuggingFace，并上线 ModelScope、百度智能云与百度 AI 开放平台，是目前长文档解析领域热度最高的开源项目。

项目热度可以用"现象级"形容：第一代模型发布次日即登顶 GitHub Daily Trending 总榜、Python 榜，以及 HuggingFace 全球模型总趋势榜、多模态模型趋势榜，实现"四榜第一"；GitHub Star 5 天突破 1 万，获得图灵奖得主 Yann LeCun 的转发推荐；截至目前 HuggingFace 累计下载量突破 1000 万次，GitHub Star 约 2.89 万、Fork 约 3100。2.0 版本发布当天，模型实测一次性解析完《哈利·波特》全系列 7 部作品共 3623 页，平均每页约 1.5 秒，全程零崩溃。

### 核心速览表

| 维度 | 关键信息 |
| --- | --- |
| 项目定位 | 端到端超长文档解析模型（One-shot / Unlimited-horizon Parsing） |
| 当前版本 | Unlimited OCR 2（2026-10-27 发布），前代 v1（2026-06-22） |
| 模型规模 | 总参数约 3.34B，每 token 激活约 500M（MoE 架构），BF16 权重约 6.7GB |
| 核心创新 | v1：R-SWA 参考滑动窗口注意力；v2：SR-SWA 稀疏参考滑动窗口注意力 + 无限解码机制 |
| 上下文能力 | v1 为 32K（数十页）；v2 单请求可达 128K~768K token 预算，页数理论上不设上限 |
| 基准成绩 | MPDocBench 84.59 分、MarathonDocBench 77.06 分，均为自报 SOTA |
| 开源协议 | MIT（商用友好，权重与代码同协议） |
| 技术栈 | Python / PyTorch / Transformers / 定制 SGLang / vLLM / ms-swift（训练） |
| 官方测试硬件 | NVIDIA A800-80GB（Transformers 与 SGLang 均在此环境验证） |

**一句话总结**：这是一个用 5 亿激活参数把"整本书一次读完"变成现实的开源模型，架构创新（SR-SWA）是它的灵魂，MIT 协议 + 完整服务化方案是它对二次开发者最大的善意。

---

## 2. 项目背景与定位：OCR 迈向 3.0 时代

要理解 Unlimited-OCR 的价值，需要先看清文档解析技术的三个发展阶段。**OCR 1.0 时代**以 Tesseract、传统 PaddleOCR 检测识别管线为代表，只做单页内的"文字检测 + 识别"，不理解版面语义，多页文档只能靠人工或脚本拼接结果，跨页表格、跨页断句全部丢失。**OCR 2.0 时代**以 GOT-OCR2.0、Nougat、DeepSeek-OCR 等端到端视觉语言模型为代表，模型直接"看图说话"，一并输出文本、表格、公式与阅读顺序，版面理解能力质变；但受制于注意力机制的二次方复杂度，上下文长度有限，DeepSeek-OCR 单次前向也只能处理数十页以内的内容。**OCR 3.0 时代**则是 Unlimited-OCR 提出的愿景：通过稀疏注意力与窗口化解码，让"页数"不再是模型的约束条件，实现"Unlimited-Horizon Parsing（无限视野解析）"。

Unlimited-OCR 与 DeepSeek-OCR 有直接的血缘关系。v1 技术报告（arXiv:2606.23050）开篇即说明：以 DeepSeek-OCR 为基线，把解码器中所有注意力层替换为自研的 R-SWA（Reference Sliding Window Attention，参考滑动窗口注意力），在保持 KV cache 恒定的前提下，把单次前向可解析的页数从几页推到数十页。v2 则更进一步，把"参考滑窗"升级为"稀疏参考滑窗"（SR-SWA），并用无限解码机制拆掉显存墙。两代模型都保留了 DeepSeek-OCR 的视觉编码器与 MoE 解码器骨架，因此它是一个"站在 DeepSeek 肩膀上解决长度问题"的专项模型。

项目的命名本身就是一次社区共创的佳话。"Unlimited OCR Works" 致敬了动漫《Fate/stay night》中的经典咏唱"Unlimited Blade Works（无限剑制）"。第一代发布时，一位名叫 Snarcy 的开发者在 HuggingFace 评论区把台词改写成了 OCR 版本——"像素为身，Token 为血；不为 KV Cache 所限，不为上下文长度所缚"。百度团队将这段网友创作原样印在了 v2 技术报告引言的第一页，并署上了原作者 ID。这种社区互动气质，也是该项目能快速积累口碑的原因之一。

---

## 3. 系统架构深度拆解

### 3.1 总体架构：DeepEncoder + 3B MoE 解码器

Unlimited OCR 2 延续了 DeepSeek-OCR 的"高压缩视觉编码 + 稀疏激活语言解码"骨架，整体数据流为：**页图像 → 视觉编码器（SAM ViT-B + CLIP-L 双塔）→ 16 倍卷积压缩 → 多页 token 拼接 → MoE 解码器自回归输出结构化文本**。

- **视觉编码器（DeepEncoder 风格）**：SAM ViT-B 负责高分辨率的局部细节感知（文字笔画、表格线），CLIP-L 负责全局语义理解（版面结构、图文关系），两者串联后经一个 16 倍压缩率的卷积模块把视觉 token 数量大幅压缩。
- **token 预算**：多页模式下每页仅消耗 **273 个 image token**。这是一个非常激进的压缩率——意味着 100 页文档的视觉输入只占约 2.7 万 token，为超长文档解析提供了前提。
- **解码器（3B MoE）**：12 层混合专家网络，共 64 个路由专家 + 2 个共享专家，每个 token 激活 6 个路由专家。总参数约 3.34B，每 token 实际激活约 500M（v1 论文中该解码器被称为 DeepSeek3B-MoE-A570M）。稀疏激活让模型在保持 3B 级知识容量的同时，把单步推理成本压到 0.5B 级别——这是它能在消费级硬件上流畅运行的关键。

### 3.2 两种视觉编码模式：base 与 gundam

模型对输入图像支持两种编码配置，理解它们的差异是正确使用模型的前提：

| 模式 | 分辨率配置 | 切片方式 | 适用场景 |
| --- | --- | --- | --- |
| **base** | base_size=1024, image_size=1024，crop_mode=False | 整页编码为一张全局视图 | 多页/PDF 解析（唯一选择）、token 预算紧张时的回退 |
| **gundam** | base_size=1024, image_size=640，crop_mode=True | 动态切片（全局视图 + 高清局部瓦片） | 单页高精度解析：论文、票据、密集表格、小字号 |

gundam 模式相当于"先看全貌、再盯细节"：全局视图提供版面结构，动态切片补充高分辨率细节，因此单页识别精度更高，但 token 消耗也更大。多页推理（infer_multi）只使用 base 模式——每页一张全局视图，靠数量换长度。v2 的 infer.py 中还实现了一个自动回退逻辑：当某文档的 token 预算装不下 gundam 模式的图像 token 时，自动降级为 base 模式。

### 3.3 注意力机制的代际演进：R-SWA → SR-SWA

这是整个项目最核心的技术创新，也是两代模型改名换姓的根本原因。

**v1 的 R-SWA（参考滑动窗口注意力）**：全注意力机制下，每生成一个 token 都要回看前面所有页面的全部视觉 token，计算量随文档长度二次方增长。R-SWA 的思路是把解码器每一层的注意力拆分为"局部滑窗 + 固定数量的参考 token"，使 KV cache 在整个解码过程中保持恒定，从而在 32K 上下文内实现数十页的一次性解析。它的局限在于：视觉 token 仍然全部参与（v1 对全部页面的视觉 token 保持可见），页数继续增长时编码与注意力开销依然会膨胀，实测极限约 40 余页。

**v2 的 SR-SWA（稀疏参考滑动窗口注意力）**：借鉴人类"抄书"的工作记忆——抄书人 95% 的时间只盯当前页，只在需要核对前文时精准回翻相关页面，而不是每次都重读整本书。SR-SWA 用两个组件实现这一直觉：

1. **Page Gate（页面门控）**：为每个解码查询（query）动态筛选"相关页面"，把视觉参考范围从整份文档缩小到少数候选页，同时建模跨页位置关系；
2. **Token Sieve / Token Buckets（令牌筛 / 令牌桶）**：在候选页内按 32-token 分块打分，只保留得分最高的 top-16 个块参与注意力计算。配合 128-token 的局部窗口，最终每个解码步的视觉参考被压缩到一个**固定 512 token 的工作参考集**（128 局部 + 16×32=512 参考块）。

两个关键的设计细节值得二开者注意：其一，这套筛选器的路由打分**直接参与前向计算**，OCR 转录的梯度可以端到端回传——也就是说"该看哪一页、抓哪一块"是模型在训练中自己学会的，而不是靠蒸馏模仿出来的固定规则；其二，每步计算的视觉信息量是**常数**，与文档是 10 页还是 5000 页无关，这正是"单步解码延迟与显存近乎恒定"的数学来源。

**代价与取舍**：稀疏化意味着模型不再保留全文档的全局信息。官方在技术报告中坦诚承认，这限制了标题层级（heading hierarchy）等文档级结构理解能力——MarathonDocBench 上标题层级子项仅 48.2 分，是所有子项中的最低分。这是架构层面 "长度的自由" 与 "全局视野" 之间的权衡。

### 3.4 Unlimited-Horizon Decoding：无限解码机制

SR-SWA 把单步计算锁定为常数之后，百度配套推出了"无限解码机制"解决显存随页数累积的问题。它像一场无缝交接的接力跑：

- **页窗口调度**：超长文档被切分为固定页数的窗口（SGLang 部署默认 10 页/窗），窗口间依序推进；
- **边界页 KV 复用**：相邻窗口共享一个"交接页（Boundary Page）"，上一窗口末页的 KV cache 直接复用到下一窗口，确保翻页处的断句、跨页表格不断流；
- **窗口显存物理释放**：当前窗口转录完毕后，历史窗口的 KV cache 立即从显存中释放，"边读边扔，轻装上阵"。

实测效果（官方 MarathonDocBench 测试环境）：单序列峰值 KV 显存从标准解码的 **47.00GB 压至 1.77GB，降幅 96.2%**；16 路并发吞吐从 360 TPS 提升至 **1315 TPS（3.65 倍）**；百页文档场景单步解码速度较 Full Attention 提升 **7.0 倍**，较 v1 的 R-SWA 提升 **5.6 倍**。需要客观指出：图像编码时间与总时长仍随页数线性增长（编码每页都要算），"恒定"的是单步解码延迟与 KV 显存，不是总耗时。

### 3.5 训练配置参考

官方披露的训练规模可作为微调成本的参照系：约 **200 万多页文档 + 1000 万单页文档**，在 **25 个节点 × 8 张 NVIDIA H800（共 200 卡）**上完成训练。对个人与中小团队而言，全参训练不现实，但 9.2 节会介绍基于 ms-swift 的 LoRA 轻量微调路线，社区已有可行的低成本实践。

---
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
- **16GB 卡（RTX 5080 等）**：Transformers 路线的单页与顺序多页解析可行（显存预算约 10-13GB），但需注意 Blackwell 架构的环境适配；SGLang 服务化不建议尝试。显存预算与环境自检详见 7.4 节。
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

### 7.4 消费级显卡部署实践（RTX 5080 16GB 案例）

7.1 节将 24GB 视为消费级下限，本节把边界进一步下探，以 **RTX 5080（16GB GDDR7，Blackwell 架构，计算能力 sm_120）** 为案例给出 16GB 显存的可行性边界。结论先行：**Transformers 路线的单页解析与顺序多页解析可以在 16GB 上稳定运行；SGLang 服务化路线则不建议在 16GB 卡上尝试。** 该结论的依据来自显存预算拆解与架构适配两方面，下面分别展开。

**单页顺序推理的显存预算拆解（BF16）**：

| 显存项 | 估算占用 | 说明 |
| --- | --- | --- |
| 模型权重（BF16） | 约 6.7GB | 3.34B 参数全量驻留；MoE 每 token 仅激活 0.5B，但权重不能只装激活部分 |
| 视觉编码器（SAM ViT-B + CLIP-L） | 约 0.5-1GB | 双塔编码器参数量很小 |
| 视觉编码激活值 | 约 1-2GB | 高分辨率输入与 gundam 动态切片时的中间张量峰值 |
| KV Cache | 百 MB 级 | SR-SWA 每步固定 512 token 工作参考集，KV 不随页数线性累积 |
| CUDA 上下文与框架开销 | 约 1-1.5GB | PyTorch、cuBLAS 工作区与分配器元数据 |
| **峰值合计** | **约 10-13GB** | batch=1、页面顺序处理的典型区间 |

16GB 装得下单请求推理，但余量不算富裕：batch=1 的顺序处理没有问题，2-4 路并发则需要精打细算甚至频繁触发显存回收。这份余量本身就是 v2 架构红利的直接体现——若 KV 像传统全注意力那样随页数线性增长，数十页上下文在 16GB 上根本无法容纳。上表为估算区间，实际峰值与页图分辨率、切片数量相关，首次跑批时建议用 `nvidia-smi` 全程观察。

**坑一：Blackwell 架构适配（比显存更先遇到）**。5080 属于 Blackwell 架构（计算能力 sm_120），要求驱动 570 系列以上、PyTorch 构建的 CUDA 版本不低于 12.8——自 torch 2.7.0+cu128 构建起才包含 sm_120 内核。官方 v2 测试栈（torch 2.13.0）不存在此问题，但若沿用旧构建（cu118/cu121）或来源不明的镜像分发，推理时会直接报错 `NVIDIA GeForce RTX 5080 with CUDA capability sm_120 is not compatible with the current PyTorch installation`。因此装完环境后必须先做内核自检：

```bash
# Blackwell 消费卡环境自检（两步都要过）
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import torch; print(torch.cuda.get_arch_list())"
# 输出必须包含 sm_120，否则报 no kernel image / not compatible 错误
```

**坑二：FlashAttention-2 慎装**。FA2 对消费级 Blackwell 的支持明显滞后于数据中心卡，直接 pip 安装容易编译失败或运行崩溃，社区反馈包括崩溃与速度异常。务实做法是加载模型时显式指定 PyTorch 原生 SDPA 注意力实现——它对 sm_120 的覆盖由 PyTorch 官方维护，稳定性远好于自行编译 flash-attn：

```python
model = AutoModel.from_pretrained(model_name, trust_remote_code=True,
                                  torch_dtype=torch.bfloat16,
                                  attn_implementation="sdpa").eval().cuda()
```

**运行参数三条建议**：

1. 设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，开启可扩展分段分配以降低碎片导致的假性 OOM，小显存卡必开（与 7.3 节 SGLang 启动脚本中同一变量的作用相同）；
2. 保持 batch=1、页面顺序处理，速度按官方长文档实测的约 1.5 秒/页量级预估即可，学习与单文档场景完全够用；
3. 不必担心长文档上限——v1 的 32K 上下文约 40 页上限由滑动窗口机制决定，与显存无关，16GB 卡同样能顺序跑完整本长文档。

**16GB 显卡的能力边界**：

| 场景 | 可行性 | 原因与说明 |
| --- | --- | --- |
| 单页解析（gundam 模式） | 可行 | 峰值约 10-13GB，16GB 装得下且有余量 |
| 长文档顺序解析（base 模式） | 可行 | KV 不随页数累积，耗时按约 1.5 秒/页线性预估 |
| SGLang 服务化 | 不建议 | 官方在 A800-80GB 测试，需定制 wheel 与一组环境变量调优；16GB 极易触发 462MiB SAM 偏置 OOM 与 KV 池预分配失败 |
| 16 路级高并发 / 批量压测 | 不可行 | 并发请求槽、CUDA Graph 与激活显存成倍增长，16GB 无法容纳 |
| INT8 权重量化（torchao 等） | 可试验 | 权重可压至约 3.5GB，但 MoE 量化兼容性无官方保证，精度需按第 8 章基准自行回归 |

**定位建议**：把 16GB 卡当作学习与单文档工作机最为合适——跑通 7.2 节路线、验证 4.3 节的结构化输出、调试后处理脚本、解析私有文档都是舒适区。若目标是 9.1 节的服务化封装或高并发生产部署，建议租用 A800/H800 或升级到 24GB 以上设备，再把 SGLang 路线纳入计划。

### 7.5 调用端：OpenAI 兼容 API

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

### 7.6 常见坑清单（部署前必读）

1. **462MiB OOM**：SAM 相对位置偏置默认一次申请约 462MiB，显存紧张时必设 `SGLANG_OPT_UOCR_SAM_REL_POS_BIAS_MB=64`（官方确认数值等价、峰值更低）。
2. **cuda-graph 桶不匹配**：`--cuda-graph-bs-prefill` 最大桶若小于 `--max-prefill-tokens`，超限批次静默回退 eager 模式，吞吐骤降且无报错。
3. **full_prefill_max_req 默认值陷阱**：默认 8，16/32 并发时会回退 eager，必须显式配到实际并发档位。
4. **页窗口与跨页质量**：窗口越小峰值显存越低，但窗口边界的跨页断句/续表质量会受损——精度敏感场景优先调大窗口而不是调小。
5. **max_tokens 语义**：是"全文档总输出预算"而非单页，多页文档必须按 `tokens_per_page × n_pages` 给足。
6. **skip_special_tokens 必须 False**：否则 `<PAGE>` 标记被吞掉，页对齐与后处理全部失效。
7. **trust_remote_code=True**：会执行模型仓库中的自定义 Python 代码，生产环境建议固定 revision 并审计代码。
8. **wheel 版本锁定**：定制 SGLang 与上游独立演进，升级前先看仓库 README 是否有配套说明。

---
## 8. 基准成绩与详细横向对比

### 8.1 两大长文档基准：整体得分排名

**MPDocBench**（数十页级多页文档，公开基准，433 份文档 3246 页）：

| 排名 | 模型 | 综合得分 |
| --- | --- | --- |
| 1 | **Unlimited OCR 2** | **84.59** |
| 2 | Qwen3.8-Max（闭源通用） | 78.30 |
| 3 | Qwen-3.8-Flash-Next | 77.36 |
| 4 | Gemini-3.8-Flash | 76.25 |
| 5 | Gemini-3.1-Pro Preview | 71.79 |
| 6 | GLM-5.3-Flash | 68.02 |
| 7 | DeepSeek-4.1-Flash | 66.25 |
| 8 | Kimi-K3 | 60.77 |

**MarathonDocBench**（百页级极限基准，百度自建：100 份 103-300 页文档、20.5 万标注版面元素）：

| 排名 | 模型 | 综合得分 |
| --- | --- | --- |
| 1 | **Unlimited OCR 2** | **77.06** |
| 2 | Kimi-K3 | 33.74 |
| 3 | Gemini-3.8-Flash | 28.61 |
| 4 | Qwen-3.8-Flash-Next | 27.83 |
| 5 | Qwen3.8-Max | 22.93 |
| 6 | MiniMax-M3 | 17.97 |
| 7 | DeepSeek-4.1-Flash | 12.34 |
| 8 | GLM-5.3-Flash | 7.51 |

百页级场景的差距最能说明架构价值：第二名 Kimi-K3 只有 33.74 分，通用旗舰 Qwen3.8-Max 从 78.3 分（MPDoc）跳水至 22.9 分，而 Unlimited OCR 2 只下降 7.5 分——准确率曲线几乎不随页数衰减，这正是 SR-SWA "单步计算恒定"的直接体现。前代 v1 因 40 页输入上限在该基准记 0 分。

### 8.2 子项得分与能力画像

| 能力子项 | MPDocBench | MarathonDocBench | 点评 |
| --- | --- | --- | --- |
| 文本内容 | 94.8 | 89.6 | 核心基本功扎实 |
| 公式（LaTeX） | 89.4 | 87.9 | 论文级公式还原可靠 |
| 阅读顺序 | 92.3 | 85.3 | 多栏/折页版面处理好 |
| 表格（TEDS） | 87.8 | 74.8 | 单页表格强，百页跨页表格衰减 |
| 标题层级 | 60.1 | 48.2 | **全子项最低分，官方承认的架构短板** |

### 8.3 开源 OCR / 文档解析模型横向对比

| 模型 | 开发方 | 参数量 | 协议 | 长文档机制 | 长文档成绩 | 单页精度（OmniDocBench v1.6） | 服务方案 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Unlimited OCR 2** | 百度 | 3.34B | **MIT** | SR-SWA + 无限解码 | MPDoc 84.6 / Marathon 77.1（自报） | 94.02 | Transformers / 定制 SGLang / vLLM |
| Unlimited-OCR (v1) | 百度 | 3.34B | MIT | R-SWA，40 页上限 | MPDoc 61.9 / Marathon 0.0 | — | Transformers / vLLM / SGLang |
| DeepSeek-OCR-2 | DeepSeek | 3.39B | Apache 2.0 | 上下文窗口内多页 | 未报告同基准 | — | Transformers / SGLang |
| PaddleOCR-VL-1.6 | 百度 PaddlePaddle | 0.96B | Apache 2.0 | 单页为主 | — | **96.34** | PaddleOCR / Transformers |
| TeleOCR | — | ~1B 级 | — | — | — | **96.91（该榜第一）** | — |
| OvisOCR2 | — | ~1.2B 级 | — | — | — | 96.47 | — |
| GOT-OCR2.0 | stepfun 等 | ~0.6B | — | 单页 | — | 较早期 SOTA | Transformers |
| 传统管线（Tesseract 等） | 开源社区 | — | Apache/MIT | 无 | 无版面理解 | 依赖预处理 | 任意 |

### 8.4 客观性提醒（阅读榜单的正确姿势）

第一，MarathonDocBench 是百度自建基准，且两大基准上的分数均为**官方自报**（以图表形式发布），截至 2026-10-27 尚无独立第三方复现，采信时应保留折扣。第二，对比对象多为通用对话大模型而非专用 OCR 工具，"领先通用模型"本身合乎预期；而在**单页精度**的公共基准 OmniDocBench v1.6 上，Unlimited OCR 2 的 94.02 分并不占榜首——TeleOCR（96.91）、OvisOCR2（96.47）、PaddleOCR-VL-1.6（96.34）都更高，这说明"长文档王"不等于"全能王"。第三，闭源通用模型在自家工具链（如 Gemini 的 Document AI）中可能有未体现在该榜的版面输出能力。

### 8.5 选型结论

- **百页级超长文档一次性解析**（整书、招股书、标书、合订档案）：Unlimited OCR 2 是当前开源阵营唯一解，MIT 协议加持下几乎没有对手；
- **单页高精度、高吞吐场景**（票据批量识别、单页合同抽取）：PaddleOCR-VL 1.6 等专用小模型精度更高、硬件要求更低，应优先对比测试；
- **深度版面分析与版式还原**：注意其标题层级短板，关键业务建议叠加规则后处理或微调补偿；
- **多页中等长度（10-40 页）**：v1、DeepSeek-OCR-2 与 v2 均可覆盖，v2 的跨页合并质量最好。

---

## 9. 二次开发方向详解（四大路线）

### 9.1 路线一：服务化封装

SGLang 定制 wheel 已经提供了 OpenAI 兼容的高性能底座，二开的工作量集中在"业务 API 网关"层：

- **架构建议**：`业务方 → 自建 REST 网关（FastAPI）→ 任务队列（Redis/RabbitMQ）→ SGLang 实例池 → 结果存储（对象存储 + DB）`。OCR 任务天然是重负载长任务，强烈建议异步化：网关受理即返回任务 ID，解析完成后回调或轮询取结果。
- **网关层必备能力**：API Key 鉴权与配额、文件类型/页数/大小校验、页图预处理（PyMuPDF 转图 + DPI 控制）、max_tokens 预算计算（可直接移植 infer.py 的 plan() 逻辑）、防复读参数默认值固化、失败重试与 refit（infer.py 的重试模式可照抄）。
- **弹性与调度**：多实例前置 Nginx/LB；按文档页数路由——短文档走 24GB 卡小实例、长文档走 80GB 卡大实例；K8s HPA 按队列深度扩缩容。
- **可观测性**：重点监控每页吞吐（tokens/s/page）、首 token 延迟、每文档总耗时、OOM 与 eager 回退次数（cuda-graph 配置失效的信号）。
- **容器化**：模型卡列出 Docker Model Runner 部署路径；自建镜像时注意把 wheel/ 里的定制 SGLang 与环境变量固化进镜像，配置漂移是这类服务最常见的翻车点。

### 9.2 路线二：模型微调

官方生态已打通训练链路（2026-07-21 起支持 ms-swift），这是四个方向中技术门槛最高、但护城河也最高的：

- **适用场景**：行业票据（发票/银行回单/报关单）、手写体、竖排古籍、特定行业版式（病历、检测报告）、以及补短板——用标题层级标注数据改善 heading 还原弱项。
- **数据规范**：按模型原生输出格式构造标注——页图 + `<BLOCK>label<LEVELn>[x1,y1,x2,y2]<FORK>内容` 序列。现有标注资产（LabelMe/COCO 版面标注、Excel 结构化数据）需要写转换器对齐到该格式；utils/process_unlimitedOCR2.py 的解析正则可以反向复用。
- **训练策略梯度**：LoRA/QLoRA（单卡 24-48GB 可试，优先调低 vision 塔学习率）→ 部分层全参 → 全参（参照官方 200×H800 量级，仅建议机构级投入）。多页长序列训练建议从单页样本起步验证格式正确性，再混入多页样本。
- **工程注意**：模型代码经 trust_remote_code 加载，微调框架需要能识别自定义模型类（ms-swift 已官方适配；LLaMA-Factory 等需自行验证）；SR-SWA 的路由打分参与前向计算，改动注意力配置时要确认梯度通路完整。
- **评估闭环**：建议自建 200-500 页的黄金测试集，按 8.2 节的五个子项维度打分，避免只看整体分。

### 9.3 路线三：业务集成

结构化输出 + 坐标是这个模型给下游应用的独特红利，远比"拿到一堆纯文本"有价值：

- **RAG 知识库**：不要把整本 Markdown 粗暴切片。利用标题层级（LEVELn）做层级化分块（章节 → 小节 → 段落），表格转 HTML/Markdown 保留单元格语义，图片用 `pageN_x1_y1_x2_y2.jpg` 引用并在检索结果中回显原图裁剪——检索命中可精确到原文坐标，这是通用大模型 PDF 管线做不到的溯源粒度。
- **文档问答与审计**：bbox 坐标直接支撑"答案在原文第几页第几块"的高亮展示；金融/法务场景可做识别结果与原件的坐标级比对审计。
- **工作流嵌入**：审批流（合同关键页抽取）、档案数字化（整卷扫描件一次入库）、舆情/研报监控（每日百页级 PDF 自动转结构化数据）。utils/ 的 convert() 可直接产出 Markdown，再按业务需要二次加工。
- **与通用大模型协同**：让 Unlimited-OCR 负责"看"（像素 → 结构化事实），通用 LLM 负责"想"（基于结构化文本做摘要/抽取/问答）——分工架构同时规避了通用大模型的幻觉与长文档失忆。

### 9.4 路线四：产品化开发

- **前端界面**：参考官方 HF Spaces Demo 的形态——上传/拖拽、逐页缩略图、解析结果与原图双栏联动、点击文本块在原图上高亮对应 bbox（0-1000 坐标乘以页面尺寸即可换算）。React/Vue + canvas 叠加层即可实现。
- **批量处理工具**：infer.py 已经是半个产品——补上断点续跑、结果缓存、失败文档单独重试队列、GUI/CLI 包装，就是团队内部的批量数字化工具。
- **私有化 SaaS 方向**：整书数字化（出版社/图书馆）、招股书/年报结构化（金融数据服务商）、标书解析（招投标工具链）、古籍数字化（配合微调）。5 亿激活参数意味着单卡即可承载多路并发，边际成本显著低于"通用旗舰模型 + 百万 token 账单"的云 API 路线。
- **商业化注意**：MIT 协议允许闭源商用与二次销售，无需开源衍生品；但输出内容的版权合规、以及"识别结果即事实"场景下的人工复核机制，需要产品层面自行设计。

---

## 10. 局限性与风险提示

**官方自述的两大局限**：其一，训练数据"多样性仍然有限"，长尾版式、特殊行业文档可能表现不稳定；其二，"当前稀疏注意力机制不保留全局文档信息，限制了标题层级等文档级结构的理解"——MarathonDocBench 标题层级仅 48.2 分是最直接的印证。依赖文档大纲结构的下游应用（如自动生成全书目录树）需要对输出做二次规整或微调补偿。

**评测公信力风险**：MarathonDocBench 为自建基准，所有对比分数为官方自报且以图片形式发布，截至报告日无独立复现；对比对象以通用对话模型为主。选型决策前，务必用**自己的真实文档**跑 PoC 验证，而不是直接采信榜单。

**工程与运维风险**：定制 SGLang wheel 意味着与上游社区版本脱钩，安全补丁与新特性跟进依赖百度仓库的更新节奏；trust_remote_code 存在供应链安全语义，生产环境应固定 revision 并做代码审计；长文档总耗时随页数线性增长，前端产品需要设计进度反馈与失败续跑，避免用户误以为服务卡死。

**许可与合规**：MIT 协议商用友好，权重同协议，这是相对部分"权重单独授权"项目（如部分 DeepSeek 系列的附加条款争议）的明确优势；但把识别结果用于决策时，请保留人工复核通道——OCR 永远存在错识率，尤其是印章遮挡、手写批注、超小字号区域。

---

## 11. 上手路线图与学习建议

**五步进阶路线**（对应 NVIDIA GPU 环境）：

1. **第 1 步 · 跑通单页**：24GB+ 显卡按 7.2 节装 Transformers 环境（16GB 卡同样可行，显存预算与环境适配见 7.4 节），下载权重（约 6.7GB），对一张论文截图跑 `model.infer()`（gundam 模式），观察输出的 `<BLOCK>` 结构与 bbox；
2. **第 2 步 · 多页与后处理**：找一份 20-50 页 PDF，PyMuPDF 转 300dpi 页图，`model.infer_multi()` 一次解析，再跑 `utils/process_unlimitedOCR2.py` 得到 Markdown，重点检查跨页表格合并质量；
3. **第 3 步 · 服务化压测**：按 7.3 节起 SGLang 服务（80GB 卡可直接套官方参数，24GB 卡先降配试验），用 OpenAI SDK 写并发客户端压测，记录吞吐与首 token 延迟基线；
4. **第 4 步 · 领域微调**：收集 200-1000 张自有场景页图，按 9.2 节格式构造标注，用 ms-swift 跑 LoRA，对比微调前后的子项得分；
5. **第 5 步 · 业务落地**：按 9.1/9.3 节搭网关与队列，选一个真实业务流（如周报归档、合同入库）小范围上线，迭代后处理规则。

**官方资源清单**：

| 资源 | 地址 |
| --- | --- |
| GitHub 仓库 | https://github.com/baidu/Unlimited-OCR |
| HuggingFace 模型（v2） | https://huggingface.co/baidu/Unlimited-OCR-2 |
| HuggingFace 模型（v1） | https://huggingface.co/baidu/Unlimited-OCR |
| 技术报告 | arXiv:2606.23050（仓库内 Unlimited-OCR-2.pdf 为 v2 报告） |
| ModelScope | 模型页同步上线（2026-06-23 起） |
| 在线 Demo | HuggingFace Spaces（README 内入口） |
| 国内平台 | 百度智能云 / 百度 AI 开放平台（2026-07-03 起上线） |
| 训练框架 | ms-swift（GitHub: modelscope/ms-swift） |

**结语**：Unlimited-OCR 用一个 3.34B 总参、5 亿激活的轻量模型，把"页数"从文档解析的约束条件里彻底移除，其价值不仅在于榜单分数，更在于它把"结构化输出 + 坐标溯源 + 恒定解码成本"打包成了一个对二次开发极其友好的 MIT 基座。对于要做私有化文档智能的开发者，它是 2026 年最值得投入时间的开源项目之一；而读懂 SR-SWA 与页窗口调度这两块"心脏"，是把它用好的第一步。
