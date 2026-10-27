# docs 目录索引

本目录是仓库的唯一文档沉淀处（对应 `docs/github-collaboration-rules.md` 规则 4），所有分析报告、图表与协作记录均在此提交推送。

## 文档清单

| 文件 | 说明 |
| --- | --- |
| `github-collaboration-rules.md` | GitHub 协作规则 v1.1：7 条规则总览与细则、沙箱准确路径、凭证安全约定、接入状态 |
| `../hf-model-code/` | 上游 HuggingFace 建模源码与配置（10 个 .py + 5 个 JSON，不含权重，MIT）来源与用法见其内 README |
| `prd/pdf-ocr-prd.md` | PRD：基于 Unlimited OCR 的 PDF 结构化解析管线（范围、功能需求、精确依赖版本、里程碑） |
| `prd/deployment.md` | 配套部署文档：双路线完整部署命令、客户端代码、调优速查与验收清单 |
| `prd/env-compat-eval.md` | 用户环境依赖版本适配评估：实测 pip list 对照基线逐项判定（transformers 5.x 阻断依据等） |
| `sr-swa-source-walkthrough.md` | SR-SWA 源码导读：稀疏参考滑动窗口注意力的类分工、四大机制与二开切入点 |
| `upstream-source-code-analysis.md` | 上游源代码分布调研：MIT 完整开源、源码托管于 HuggingFace/ModelScope 的实证与清单 |
| `Unlimited-OCR-深度拆解报告.md` | 百度 Unlimited-OCR 开源仓库深度拆解报告（约 9060 字，含架构、部署、二开方向、竞品对比、RTX 5080 部署实践） |
| `百度Unlimited-OCR深度拆解报告.docx` | 上报告的 Word 版本（26 页，TOC/页脚/postcheck 9/9 通过） |
| `charts/benchmark_ranking.png` | 综合评测基准排名图 |
| `charts/radar_subscores.png` | 各维度子项得分雷达图 |

## 约定

- 此后新生成的报告类交付物直接写入本目录并随代码一起提交推送；
- 图表类附件统一放 `charts/` 子目录；
- 文件命名：分析记录用小写英文加连字符（如 `benchmark-notes.md`），正式报告可保留中文名以便阅读。
