# docs/archive — 历史版本文档归档

本目录存放已被新版本取代的文档快照，仅供追溯与对照，不再随需求演进维护。所有条目均为 git mv 归档，历史可追溯。

| 文件 | 被取代原因 | 取代者 |
| --- | --- | --- |
| `pdf-ocr-prd-v1.0.md` | 依赖版本基线（torch 2.13.0 等）被 2026-10-28 版本定稿取代；推理路线含 SGLang | `../prd/pdf-ocr-prd.md`（v1.1） |
| `deployment-v1.0.md` | 双路线部署文档，SGLang 路线移出本期；环境管理由 uv 改为 conda | `../prd/deployment.md`（v1.1） |

## 路线 B（SGLang）归档说明

v1.1 依据 `docs/prd/env-compat-eval.md` 的评估结论，将依赖版本定稿为 torch 2.11.0 栈；而路线 B 依赖的定制 SGLang wheel（`sglang-0.5.20.dev10`，仓库 `wheel/` 目录仍保留）与 torch 2.13.0 强绑定，两条基线无法共存，故将路线 B 整体移出本期并随 v1.0 文档归档。

**恢复条件**：出现高并发在线服务化需求，且确认该 wheel 与届时 torch 栈兼容（或上游发布新适配 wheel）。恢复时以 `deployment-v1.0.md` §3 为起点重新验证，勿直接复活本文档。
