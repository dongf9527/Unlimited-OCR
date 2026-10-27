# Worklog

---
Task ID: 1
Agent: main (Z.ai)
Task: 调研 baidu/Unlimited-OCR 仓库（架构/技术栈/功能/部署/二开方向），产出深度拆解报告（Word + Markdown）

Work Log:
- AskUserQuestion 澄清需求：Word报告 + 深度拆解(5000字+) + 二开四方向全覆盖 + NVIDIA GPU 部署 + 自己学习 + 详细竞品对比 + 备注"输出文档为markdown格式"（故同时交付 .md 源文件）
- 加载 docx skill（SKILL.md 已读，路由文件待读）
- 克隆仓库到 /home/z/my-project/research/Unlimited-OCR（assets/wheel/pdf 为 LFS 指针，代码完整）
- 走读 infer.py（204行）：基准上下文 MPDocBench=131072 / MarathonDocBench=786432；plan() token 预算规划；gundam→base 回退；服务端 refit；DeepseekOCRNoRepeatNGramLogitProcessor（ngram=35, window 128/1024）
- 走读 utils/process_unlimitedOCR2.py（380行）+ utils/README.md：模型原生输出结构化版面块（<PAGE><BLOCK>label<LEVELn>[bbox]<FORK>），18种标签 + cross_/append_ 跨页链 + <MERGE>；后处理做 union-find 跨页表格合并、表头去重、坐标 0-1000 裁剪
- git show 读取 v1 README：v1 基于 python3.12/CUDA12.9/torch2.10/transformers4.57.1，max_length=32768，gundam(base 1024+image 640 切片)/base(1024) 模式，vLLM+SGLang 部署，批量 infer.py 自动起服务
- WebFetch GitHub README（v2）：SR-SWA、SGLang 部署全量参数、评测脚本用法、MIT、28.9k star
- WebFetch arXiv 2606.23050（v1论文）：以 DeepSeek-OCR 为 baseline，解码器全部注意力层替换为 R-SWA，恒定 KV cache，32K 上下文一次前向解析数十页
- WebFetch HF 模型卡（v1+v2）：3.34B 参数 BF16≈6.7GB，max position 131072，A800-80GB 测试，支持 ms-swift 训练 / vLLM / Docker Model Runner
- WebSearch 竞品与实测数据：MPDocBench 排名（UOCR2 84.59 > Qwen3.8-Max 78.30 > …）；MarathonDocBench（77.06 >> Kimi-K3 33.74）；子项分（Text 94.8/89.6, Formula 89.4/87.9, ReadOrder 92.3/85.3, Tables 87.8/74.8, Heading 60.1/48.2）；架构细节（SAM ViT-B + CLIP-L → 12层 MoE 64专家/6激活/2共享，273 image tokens/页）；SR-SWA = 128 局部窗口 + top-16×32 token 块 = 512 工作参考集；KV 47GB→1.77GB(-96.2%)，TPS 360→1315；训练 200×H800；局限（标题层级弱、自报基准）；单页基准 OmniDocBench v1.6 94.02 不敌 TeleOCR 96.91/OvisOCR2 96.47/PaddleOCR-VL-1.6 96.34

Stage Summary:
- 调研完成，信息源：仓库代码 + v1/v2 README + arXiv 论文 + HF 模型卡 + 多家媒体实测报道
- 下一步：产出 Markdown 主文档（download/）→ matplotlib 基准图表 → 按 docx skill 文件链生成 Word → postcheck → 交付

---
Task ID: 2
Agent: main (Z.ai)
Task: 产出交付物（Markdown 源文档 + 基准图表 + Word 深度报告）

Work Log:
- 按大纲分三部分撰写报告正文（scripts/report_part1-3.md），合并为 download/Unlimited-OCR-深度拆解报告.md（23433 字符 / 中文 8001 字，满足深度拆解 5000+ 要求）
- scripts/gen_charts.py 生成两张 matplotlib 数据图（Noto Serif SC 字体、constrained_layout）：download/charts/benchmark_ranking.png（双基准横向柱状图）、radar_subscores.png（五维雷达图），均经视觉校验修复标签重叠
- 完整阅读 docx 技能文件链：SKILL.md → routes/create.md → references/design-system.md（R1 封面配方/DM-1 配色/calcTitleLayout/calcCoverSpacing/表格样式/排版规则）→ common-rules.md（Profile A 字体/WPS 兼容/页码五区）→ toc.md（TOC 三步流程/三节页码架构）→ docx-js-core.md → scenes/report.md；并核查 postcheck.py 源码确认检查逻辑（行距一致性、图片溢出、字体白名单、TOC、双分页判定）
- scripts/docx_gen/：generate.js（R1 封面 + 三节架构 + 块渲染器：h1/h2/h3/p/code/table/img/bullets/olist，内容与渲染分离）；content_part1-3.js（回模板字符串规避引号转义问题）；postprocess_footers.py（删空 pgNumType + 按节映射修补页脚 PAGE 域：第2节 ROMAN / 第3节 arabic）
- 生成流程：node generate.js → add_toc_placeholders.py --auto（34 条条目，exit 0）→ postprocess_footers.py → postcheck.py 9/9 通过（0 错误 0 警告）
- 修复两处问题：表 8 首行单元格分段数组少包一层括号（node 运行时报错）；TOC 域段落内 docx-js 残留 hyperlink 样例条目导致目录首条重复（fix_toc_dup.py 手术移除，保留 34 条统一占位条目）
- LibreOffice 转 PDF 渲染验证：封面（R1 深青科技风）✓、目录两页（罗马页码 I/II）✓、正文（SimSun/SimHei/黑体标题/表格斑马纹/代码块浅灰底/图表嵌入）✓，共 24 页

Stage Summary:
- 最终交付物（/home/z/my-project/download/）：
  1. 百度Unlimited-OCR深度拆解报告.docx（24 页 Word 深度报告，postcheck 9/9 通过）
  2. Unlimited-OCR-深度拆解报告.md（Markdown 源文档，中文 8001 字）
  3. charts/benchmark_ranking.png + charts/radar_subscores.png（可复用图表资产）
- 可恢复脚本：scripts/gen_charts.py、scripts/report_part*.md、scripts/docx_gen/*（改内容只需编辑 content_part*.js 后重跑生成链）
