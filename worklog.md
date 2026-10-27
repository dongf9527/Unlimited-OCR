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

---
Task ID: 3
Agent: main (Z.ai)
Task: 应用户追问（RTX 5080 16GB 能否部署），在报告第 7 章新增"消费级显卡部署实践"一节，同步 Word 与 Markdown

Work Log:
- 回答用户"5080 16G 能否部署"：结论为 Transformers 路线单页/顺序多页可行（峰值约 10-13GB），SGLang 服务化不建议；WebSearch 核实 RTX 5080 为 Blackwell sm_120，需驱动 570+ / CUDA 12.8+ / torch 2.7.0+cu128 起才有 sm_120 内核，FA2 对消费级 Blackwell 支持滞后建议用 SDPA
- 定位插入点：第 7 章原 7.4/7.5 之后无交叉引用，插入为新 7.4 并顺延旧 7.4→7.5、7.5→7.6；新增 2 张表使 part3 表 5-9 顺延为表 7-11（正文无表号引用，安全）
- Markdown（download/Unlimited-OCR-深度拆解报告.md）：插入 7.4 全节（引言 + 表：显存预算拆解 + 预算解读 + 坑一 Blackwell 自检代码 + 坑二 FA2/SDPA 代码 + 运行参数三条 + 表：16GB 能力边界 + 定位建议），7.1 务实建议补 16GB 条目，第 11 章第 1 步补 16GB 提示；中文字数 8001→9060
- content_part2.js：7.1 建议段补 16GB 句；插入 7.4 全部块（h2/p/table×2/code×2）；标题顺延 7.5/7.6；content_part3.js：5 张表 cap 顺延 + 第 11 章第 1 步更新
- 修复 1 处运行时错误：表 5"峰值合计"行加粗单元格未按 [["文本",1]] 双层嵌套（与既往表 8 同类问题）
- 重跑生成链：node generate.js → add_toc_placeholders --auto（35 条，+1）→ fix_toc_dup.py（移除 1 条残留 hyperlink）→ postprocess_footers.py（3 节 ROMAN/arabic 正常）→ postcheck.py 9/9 通过（0 错 0 警）
- LibreOffice 转 PDF 验证：总页数 24→26；7.4 实际渲染于正文 12-14 页，表 5/表 6、代码块、加粗行、页脚页码均正常；TOC 占位页码为预估值属既有机制（Word 内"更新域"刷新）

Stage Summary:
- 交付物已更新（/home/z/my-project/download/）：
  1. 百度Unlimited-OCR深度拆解报告.docx —— 26 页，11 章不变，第 7 章扩为 7.1-7.6，表格 9→11 张
  2. Unlimited-OCR-深度拆解报告.md —— 中文字数 9060，7.4-7.6 编号已顺延
- 可复现管线不变：编辑 scripts/docx_gen/content_part*.js 后重跑生成链即可

---
Task ID: 4
Agent: main (Z.ai)
Task: 将 GitHub 协作规则（7 条）整理为正式文档存入 docs/，占位符按沙箱实际路径填写

Work Log:
- 检查沙箱 git 环境：/home/z/my-project 已是 git 仓库（main 分支、身份已配置、远程未配置）；research/Unlimited-OCR 存在克隆带来的嵌套 .git，已在规则 2 细则中注明推送前处理方式
- 创建 docs/ 目录，写入 docs/github-collaboration-rules.md：规则总览表（7 条）+ 逐条细则（含 conventional commits 速查表、force push 等效情形、rebase 替代方案）+ 当前接入状态表；仓库位置占位符填为 /home/z/my-project/
- 按规则 1/3 执行本地小步提交（docs: 前缀），推送待使用者提供仓库地址与 token 后进行

Stage Summary:
- 新增 docs/github-collaboration-rules.md（规则文档 v1.0）
- 等待接入信息：第 1 步 GitHub 仓库地址，第 2 步访问 token；收到后配置 remote 并执行首次推送

---
Task ID: 5
Agent: main (Z.ai)
Task: 配置 GitHub 远程并完成首次推送（dongf9527/Unlimited-OCR），建立协作规则生效基线

Work Log:
- 配置 origin = https://github.com/dongf9527/Unlimited-OCR.git；ls-remote 确认远程为空仓库
- 推送前安全检查：发现 .env（DATABASE_URL）被 Initial commit 跟踪、research/Unlimited-OCR 为嵌套 gitlink、tool-results/ 为会话临时产物
- 因远程为空（无已推送历史），在本地执行 filter-branch 清洗三个路径后重写未推送历史（不涉及规则 5/6 保护范畴）；.gitignore 增补 .env、research/、tool-results/ 并以 chore: 提交
- 使用 token 构造一次性 Authorization Basic 头推送 main 成功（token 未写入 .git/config 或任何文件），建立上游跟踪；ls-remote 验证 origin/main = 本地 HEAD
- 更新 docs/github-collaboration-rules.md：接入状态表（远程/首推/清洗记录）+ 新增第四节凭证安全约定（token 不落盘、建议轮换、远程 URL 保持干净）

Stage Summary:
- main → origin/main 首推完成（d2ae0c1），协作规则文档随仓库上 GitHub
- 后续所有产出按规则 1-4 走 commit + push；token 使用一次性注入方式，无需重复提供（失效时再要新的）

---
Task ID: 5
Agent: main (Z.ai)
Task: 沙箱凭证存储机制建立、协作规则 v1.1 更新（沙箱准确路径）、download/ 文档迁移至 docs/

Work Log:
- 确认首次推送已于上轮完成（origin/main = b0eb161，与本地一致）
- 全盘检索确认旧 token 无任何残留（符合此前一次性使用约定，但 token 值已随上下文压缩不可恢复，需使用者重发）
- 新建 .credentials/（目录 700、github-token 文件 600）并加入 .gitignore，git check-ignore 验证生效
- 协作规则升级 v1.1：规则 2 写明唯一仓库准确绝对路径 /home/z/my-project/ 与各目录职责；第四章凭证约定由"一次性注入"改为"持久化存储"，含 30 天有效期管理（约 2026-11-26 到期）
- git mv 将 download/ 全部 5 项（README、md 报告、docx 报告、charts/ 两图）迁移至 docs/，download/ 目录删除；md 报告无仓库内相对图片引用，迁移零影响
- docs/README.md 重写为文档索引
- 曾出现 rename 混入 chore 提交的边界问题，因三个 commit 均未推送，按规则 5 用 mixed reset 重新按干净边界提交

Stage Summary:
- 本地新增 3 个未推送 commit：eaed9f4 (chore gitignore) / c5819d8 (docs 规则 v1.1) / 3db3219 (docs 迁移)
- 待办：使用者重新提供 token → 写入 .credentials/github-token → 立即推送
