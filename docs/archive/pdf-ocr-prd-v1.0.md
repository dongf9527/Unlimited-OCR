# PRD：基于 Unlimited OCR 的 PDF 文档结构化解析管线

版本 v1.0 ｜ 日期 2026-10-27 ｜ 负责人：待定 ｜ 状态：评审中

## 1. 背景与目标

公司内部存在大量扫描版与原生版 PDF：合同、招投标书、研报、教材、财务报表。现有处理方式是人工转录加通用 OCR 工具拼接，存在三个具体问题：表格跨页后结构断裂；双栏排版的阅读顺序错乱；导出格式不统一，下游系统（检索、核算、知识库）无法直接消费。

本项目的解决方案：以百度开源的 Unlimited OCR 2（`baidu/Unlimited-OCR-2`，3.34B 参数，MIT 协议）为核心引擎，搭建一条 PDF 到结构化数据的批处理管线，一次解析输出 Markdown / TXT / DOCX / JSON / 文本型 PDF 五种格式。

量化目标：

| 指标 | 目标值 |
| --- | --- |
| 单页解析延迟（SGLang 服务，A800-80G） | ≤ 2.5 s/页（gundam 模式） |
| 批量吞吐 | ≥ 1,000 页/小时/卡 |
| 跨页表格合并正确率（内部抽样 200 份） | ≥ 90% |
| 双栏阅读顺序正确率（内部抽样 100 份） | ≥ 95% |
| 管线整体页级失败率 | ≤ 0.5%，失败样本自动落盘待人工复核 |

## 2. 范围界定

做：

- PDF（扫描件与文本型均转图像处理）→ 页图像序列的解析
- 版面块识别：标题、正文段落、表格、图片位置的还原；双栏/多栏阅读序恢复
- 浅色水印抑制（渲染阶段预处理）
- 五种格式的导出与一致性校验
- 单机部署与批处理脚本、失败重试、断点续跑

不做（明确砍掉，降低维护面）：

- 不做 Web 界面与在线服务，仅命令行批处理
- 不做模型训练与微调，直接使用开源权重
- 不做手写体识别承诺、不做章节数据库、不做翻译
- 不做分布式调度（单卡单机；多卡靠目录切分并行）
- 不做复杂艺术字水印去除（仅浅色半透明水印，见 4.2）
- 不做除中文、英文外的语种保障

## 3. 系统架构与数据流

```
PDF 文件
  │ ① PyMuPDF 渲染（300 dpi，PNG）
  ▼
页图像序列 page_0001.png ...
  │ ② 预处理（浅色水印抑制，可选跳过）
  ▼
Unlimited OCR 推理
  ├─ 路线 A：Transformers 单机（≤ 50 页/次的低频场景）
  └─ 路线 B：SGLang 服务（OpenAI 兼容 API，批量默认路线）
  │ ③ 输出：逐页 <PAGE> 分块的 title/text/table/image JSON
  ▼
④ 后处理：process_unlimitedOCR2.py（纯标准库，~380 行）
  │  页切分对齐、标题树单根化、跨页表格 union-find 合并、表头去重
  ▼
⑤ 导出层：Markdown ｜ TXT ｜ DOCX(python-docx) ｜ JSON(schema 见 4.5) ｜ 文本型 PDF(reportlab)
  ▼
⑥ 校验器：页数一致性、表格行列完整性、产物哈希登记
```

设计原则：②③④⑤ 每一步的输入输出都落盘为中间产物（JSON 为唯一中间表示），任何一步失败只重跑该步，不整卷重算。

## 4. 功能需求

### 4.1 PDF 解析（FR-1）

- 使用 PyMuPDF（fitz）统一渲染为 PNG，分辨率 300 dpi（`fitz.Matrix(300/72)`）；扫描件与文本型 PDF 走同一通路，避免双逻辑。
- 单文件页数无上限；超过 10 页自动进入 SGLang 页窗模式（见 5.3）。
- 加密 PDF 先用 `fitz` 空密码探测，无法解密则记录失败原因并跳过，不中断批次。

### 4.2 水印抑制（FR-2）

范围限定为浅色半透明水印（常见于合同、研报，如浅灰 Logo、斜向文字）。实现分两层：

- 原生 PDF 且水印为独立对象：渲染前通过 `page.get_drawings()` 识别低透明度图形对象，渲染时跳过；
- 扫描件：预处理阶段做亮度阈值抑制——水印像素灰度显著高于前景文字（典型差值 > 60），用自适应阈值将水印区像素拉向背景色，不触碰文字笔画。

不承诺：与正文重叠的深色水印、照片内嵌水印。此类样本进入人工复核队列。该功能整体为可选项（`--no-watermark` 跳过），不影响主流程。

### 4.3 版面与双栏（FR-3）

依赖模型原生输出：单页默认 gundam 模式（动态切图），上游在 DeepSeek-OCR 路线上对多栏阅读序已做对齐训练。管线侧职责是按模型输出的块顺序拼接，不做二次几何推断。验收以 4.1 的双栏抽样指标为准；连续两版低于 90% 时，在该页追加 base 模式（整页单视图）重解析并取表格完整度较高者。

### 4.4 表格识别（FR-4）

- 模型输出 HTML 表格（`<table><tr><td>`，含 colspan/rowspan）；
- 后处理做跨页合并：相邻页表格经 union-find 分组，重复表头行去重（`process_unlimitedOCR2.py` 已实现，直接复用）；
- 导出层将 HTML 表转换为 Markdown 管道表（不含合并单元格时）、DOCX 原生表格（保留合并单元格）、JSON 内保留原始 HTML。
- 畸形表格（行列错位）不静默修复，保留原文并在 JSON 中标记 `quality_flag: "table_degraded"`。

### 4.5 输出与数据定义（FR-5）

JSON 为唯一中间表示与交付格式之一，schema 骨架：

```json
{
  "doc_id": "sha1(源文件)",
  "pages": [
    {
      "page_no": 1,
      "blocks": [
        {"type": "title|text|table|image",
         "level": 2,
         "content": "标题文本或 HTML 表格或 image 占位",
         "image_path": "images/page_0001_fig_0.png",
         "quality_flag": null}
      ]
    }
  ],
  "meta": {"model": "Unlimited-OCR-2", "pipeline_version": "1.0.0"}
}
```

- 图片块：解析阶段从页图像裁剪，落 `images/` 目录，各格式导出时以相对路径引用；
- Markdown：标题层级映射 + 管道表 + 图片相对引用；
- TXT：仅文本序列（标题降为带编号行），表格转 TSV；
- DOCX：python-docx 0.1.x 系 API 生成，标题用 Heading 样式，表格转原生表格；
- 文本型 PDF：reportlab 4.x 生成，中文注册 Noto Serif SC，可全文检索、可复制，供档案系统归档；
- 五种产物共享同一 doc_id，校验器核对页数与块数一致后写 `manifest.json`。

### 4.6 批量与容错（FR-6）

- 输入为目录，逐文件处理；产物目录结构与源目录镜像；
- 单页失败重试 2 次（gundam→base 降级各一次），仍失败则记录 `failed_pages.json`，文档继续；
- 断点续跑：以 `manifest.json` 判断已完成文档，重启后跳过；
- 全部日志结构化输出（JSONL），含每页耗时、token 数、重试次数。

## 5. 技术选型与依赖版本

### 5.1 运行环境（两路线共用）

| 组件 | 版本 | 说明 |
| --- | --- | --- |
| OS / CUDA | Ubuntu 22.04 / ≥ 12.8 | 驱动 ≥ 570 |
| Python | 3.10 | 上游验证基准版本，禁止用 3.12+ |
| torch / torchvision | 2.13.0 / 与 torch 同源 | |
| transformers | 4.52.1 | 必须，trust_remote_code 依赖此版行为 |
| huggingface-hub | 0.36.0 | |
| tokenizers | 0.21.4 | |
| addict / einops / easydict | 2.4.0 / 0.8.2 / 1.13.0 | 模型代码 import |
| pillow | ≥ 10.0 | 图像 IO |
| pymupdf | 1.27.2.2 | PDF 渲染 |
| python-docx / reportlab | ≥ 1.1 / ≥ 4.0 | DOCX 与文本型 PDF 导出 |
| opencv-python-headless | ≥ 4.9 | 水印抑制与图像裁剪 |

模型权重：`huggingface.co/baidu/Unlimited-OCR-2`，safetensors 两分片约 6.6GB，MIT。仓库内 `hf-model-code/` 已含全部建模源码与配置。

### 5.2 路线 A：Transformers（低频、小批量）

加载参数：`AutoModel.from_pretrained(..., trust_remote_code=True, use_safetensors=True, torch_dtype=torch.bfloat16).eval().cuda()`。推理入口用模型自带的 `model.infer(tokenizer, image_file=...)`（单页）与 `model.infer_multi(tokenizer, image_files=pages, ...)`（多页，prompt 自动构造为 `"<PAGE><image>"*n + "Multi page parsing."`）。

显存：80G 卡无压力；24G 卡可用 bf16 跑通（权重 6.7G + 激活）；16G 卡仅建议单页顺序处理并关闭 CUDA Graph。本路线不要求多并发，胜在零服务化成本。

### 5.3 路线 B：SGLang 服务（批量默认）

安装上游定制 wheel `sglang-0.5.20.dev10+g8d62f5fb9c-py3-none-any.whl`（仓库 `wheel/` 目录已含），另装 `fastokens==0.3.2`。服务启动命令与内存参数见配套部署文档 `docs/prd/deployment.md`。

关键采样参数（客户端请求体）：`temperature=0`、`max_tokens=8000×页数`、`custom_logit_processor` 用仓库内置 `DeepseekOCRNoRepeatNGramLogitProcessor`（`ngram_size=35`，多页 `window_size=1024`、单页 `128`、`page_token_id=128827`、`eos_token_id=1`、`max_page_tokens=20000`）。

页窗策略：`SGLANG_UOCR_DOC_GUNDAM_PAGE_WINDOW=10`，KV 尾部跨窗复用；页窗减小可降峰值显存，代价是跨页表格上下文变弱——跨页表格合并率下降超过 5 个百分点时回调至 10。

### 5.4 选型结论

生产批量一律走路线 B；路线 A 仅用于开发调试与个位数文档。不引入 vLLM（官方虽支持，但需另行验证幻觉与页窗行为，本期不做）。

## 6. 非功能需求

- 可观测：每文档一行 JSONL 汇总（页数、耗时、失败页、产物哈希），保留 90 天；
- 安全：文档不出内网，模型权重离线分发；日志不记录正文内容，只记录哈希与统计量；
- 兼容：产物 UTF-8，DOCX 兼容 Office 2016+，文本型 PDF 兼容主流阅读器全文检索；
- 资源上限：单卡同时只跑一个服务实例；`--mem-fraction-static 0.7` 起步，32 并发以上 OOM 才降到 0.65。

## 7. 里程碑

| 阶段 | 内容 | 验收标准 |
| --- | --- | --- |
| M1（3 天） | 环境搭建、模型加载、单页冒烟 | 路线 A/B 各出一份单页 Markdown |
| M2（5 天） | 管线主体：渲染→推理→后处理→JSON | 100 页混合样本文档跑通，失败率 ≤ 2% |
| M3（5 天） | 五格式导出层 + 校验器 | 五格式内容一致性抽查通过 |
| M4（3 天） | 批量与断点、水印抑制 | 1,000 页连续批处理，指标达第 1 节目标 |
| M5（2 天） | 部署文档、压测报告 | 第三人按文档可独立复现 |

## 8. 风险与对策

| 风险 | 对策 |
| --- | --- |
| 定制 SGLang wheel 仅提供 py3.10 manylinux，未来环境升级断供 | wheel 已归档至仓库 `wheel/`；锁文件固化全依赖 |
| 长文档 token 超预算截断 | 页级 `max_page_tokens=20000` 守卫 + 超限页拆分重跑 |
| 跨页表格合并错误引入事实性错误 | `table_degraded` 标记 + 抽样人工复核机制 |
| 显存峰值受并发影响波动 | 并发从 16 起步爬坡，逐级压测后再放 64 |
