# GitHub 协作规则

> 版本：v1.1 ｜ 生效日期：2026-10-27 ｜ 适用范围：本沙箱内所有代码与文档的协作、提交与推送行为

## 一、规则总览

| # | 规则 | 核心要求 |
| --- | --- | --- |
| 1 | 唯一同步方式 | 所有代码/文档改动通过 `git commit` + `git push` 推送到 GitHub，不在对话中凭空产生代码 |
| 2 | 仓库位置 | 严格使用沙箱中的根目录 `/home/z/my-project/` 作为唯一 git 仓库，不在其他目录创建 `.git` |
| 3 | 提交规范 | 小步提交，commit message 使用 conventional commits 风格（如 `feat:`、`fix:`、`docs:`） |
| 4 | 会话记录 | 分析/评估/测试记录写入 `docs/` 目录，以 Markdown 形式提交推送 |
| 5 | 禁止 force push | 绝不使用 `git push --force`；同理禁止对已推送 commit 执行 `--amend` 后再推送（等效 force，同样触犯本条）；未推送 commit 的 amend 属修正笔误常规手段，不受限 |
| 6 | Rebase 前置 | rebase 前先评估替代方案（revert/merge），必要时提前告知确认 |
| 7 | bug 修复不重命名变量 | 修复 bug 时保持原变量名，仅在 refactor 时重命名 |

## 二、规则细则

### 规则 1 ｜ 唯一同步方式

所有改动必须落到 git 提交并推送到 GitHub 远程，对话中给出的代码仅作为提交内容的预览或说明，不作为交付载体。这样做保证任何时刻 GitHub 上的仓库都是唯一的可信源，会话中断或更换环境后工作可以无损恢复。凡是需要长期留存的产出（代码、脚本、文档、记录），一律先落盘到仓库目录，再走提交推送流程。

### 规则 2 ｜ 仓库位置

沙箱内唯一 git 仓库的**准确绝对路径**为：

```text
/home/z/my-project/
```

该目录已初始化为 git 仓库（分支 `main`），远程为 `origin = https://github.com/dongf9527/Unlimited-OCR.git`。不得在沙箱内其他位置（如 `/home/z/`、`/tmp/`、任何子目录）执行 `git init` 或创建 `.git` 目录，避免多仓库并存导致改动散落、状态漂移。

仓库内固定目录职责（2026-10-27 调整）：

| 路径 | 职责 |
| --- | --- |
| `/home/z/my-project/`（根级） | 上游 `baidu/Unlimited-OCR` 完整快照（README/LICENSE/infer.py/utils/assets/wheel 等），作为二开基线 |
| `/home/z/my-project/docs/` | 全部文档沉淀：分析报告（md/docx）、图表（charts/）、目录索引与协作规则 |
| `/home/z/my-project/scripts/` | 生成与运维脚本（Python / Node / Shell），可迭代复用 |
| `/home/z/my-project/.credentials/` | git 凭证存储（gitignored，权限 700/600，绝不提交） |
| `/home/z/my-project/worklog.md` | 多智能体共享工作日志 |

Git LFS 约定：仅 `*.gif` 走 LFS（上游动图 149MB 超 GitHub 普通单文件 100MB 限制，见根级 `.gitattributes`）；pdf/png/whl 以普通 blob 提交；LFS filter 使用绝对路径 `/home/z/.local/bin/git-lfs`（二进制包安装于沙箱用户目录，不入仓库）。

历史说明：早期交付物存放在 `download/`，已于 2026-10-27 全部迁移至 `docs/`；此后新生成的报告类交付物直接写入 `docs/` 并提交推送。

### 规则 3 ｜ 提交规范

小步提交：一次 commit 只做一件事，便于回滚、review 与定位问题。commit message 使用 conventional commits 风格，格式为 `<type>: <简短描述>`，常用类型速查如下：

| type | 用途 | 示例 |
| --- | --- | --- |
| `feat` | 新功能 | `feat: 新增 5080 显存预算计算脚本` |
| `fix` | 修复缺陷 | `fix: 修正表格单元格嵌套导致的渲染报错` |
| `docs` | 文档变更 | `docs: 新增 GitHub 协作规则` |
| `refactor` | 重构（不改行为） | `refactor: 拆分报告生成脚本` |
| `test` | 测试相关 | `test: 补充页脚修补脚本用例` |
| `chore` | 构建/杂项 | `chore: 更新 .gitignore` |

描述使用中文或英文均可，但同一仓库内保持风格一致；一行能说清的不写多行，确需补充背景时在 message 正文空一行后说明。

### 规则 4 ｜ 会话记录

每次完成分析、评估、测试或问题排查后，将过程与结论整理成 Markdown 文件写入 `docs/` 目录，随代码一起提交推送。文件命名采用小写英文加连字符（如 `docs/benchmark-notes.md`、`docs/github-collaboration-rules.md`），保证跨平台兼容。记录内容应包含背景、方法、关键数据与结论，使未参与当次会话的人也能读懂来龙去脉。

### 规则 5 ｜ 禁止 force push

`git push --force` 会覆盖远程历史、可能抹掉他人或先前会话的工作，因此在任何分支上都禁止使用。与它等效的操作同样被禁止：对已推送的 commit 执行 `git commit --amend` 后再推送（无论后续是否需要 force），以及任何带有强制覆盖远程意图的推送行为。仅当 commit 尚未推送时，`--amend` 用于修正笔误属于常规手段，不受本条限制。误推送需要撤销时，使用 `git revert` 生成反向提交。

### 规则 6 ｜ Rebase 前置

rebase 会改写提交历史，执行前必须先评估更安全的替代方案：撤销错误提交优先 `git revert`，整合分叉历史优先 `git merge`。只有当 rebase 带来的历史整洁收益明确大于改写风险、且无替代方案能达成同样效果时，才考虑执行。涉及已推送的共享分支时，必须提前向使用者说明影响并获确认后才能操作。

### 规则 7 ｜ bug 修复不重命名变量

修复 bug 的 commit 中保持既有变量名不变，使 diff 聚焦于行为修正本身，review 时能一眼看出改了什么逻辑。重命名属于改善代码可读性的重构范畴，应放在独立的 `refactor:` 提交中单独进行。两者混在一起会放大 diff 噪声，掩盖真正的修复点，也增加回滚与二分定位的成本。

## 三、当前接入状态

| 项 | 状态 |
| --- | --- |
| 沙箱仓库路径 | `/home/z/my-project/`（已初始化，分支 `main`） |
| git 身份 | 已配置（user.name / user.email） |
| 远程 remote | `https://github.com/dongf9527/Unlimited-OCR.git`（origin，已配置） |
| 首次推送 | 2026-10-27 完成，`main` → `origin/main`（d2ae0c1），已建立上游跟踪 |
| 推送前清洗 | `.env`（含 DATABASE_URL）已从全部历史移除并加入 `.gitignore`；`research/`、`tool-results/` 已移出索引并忽略 |
| `docs/` 目录 | 全部文档沉淀处：报告 md/docx、charts/ 图表、README 索引及本规则文件（对应规则 4） |
| 目录迁移 | 2026-10-27 将 `download/` 下全部文档与图表迁移至 `docs/`，此后报告类交付物直接写入 `docs/` |
| 凭证存储 | `/home/z/my-project/.credentials/github-token`（gitignored，权限 600，目录 700）；2026-10-27 首次写入并推送；2026-10-28 因沙箱回滚丢失后重发重建（API 验证 HTTP 200, login: dongf9527），有效期至约 2026-11-26 |
| 上游同步 | 2026-10-27 将 `baidu/Unlimited-OCR` 快照 `0cdd787` 同步至仓库根（LFS 指针已替换为真实文件并逐个 sha256 校验）；后续增量同步复用 `scripts/sync_upstream.sh` |

## 四、凭证安全约定（v1.1 更新：持久化存储）

经使用者明确授权，访问 token 由沙箱妥善保存，供 30 天有效期内反复使用：

- **存储位置**：`/home/z/my-project/.credentials/github-token`（纯文本仅存 token 本体）；目录权限 `700`、文件权限 `600`，仅沙箱内 git 推送/拉取流程可读；
- **绝不入库**：`.credentials/` 已加入 `.gitignore`（已验证生效），token **不写入** `.git/config`、remote URL、任何脚本或提交内容；
- **使用方式**：推送/拉取时以临时 credential helper 或一次性 `Authorization` 请求头从该文件读取注入，用后不残留于进程参数或日志；
- **有效期管理**：token 有效期 30 天（自 2026-10-27 起算，约 2026-11-26 到期），到期前提醒使用者轮换；建议定期在 GitHub Settings → Developer settings 中轮换 fine-grained PAT，并保持最小授权（仅目标仓库 Contents: Read/Write）；
- **远程 URL 干净原则**：`origin` 始终保持不含凭证的干净形式，避免 token 随仓库配置泄漏。

## 五、沙箱环境风险与防御（v1.1 补充，2026-10-27）

沙箱会在会话之间清理**未被 git 跟踪**的内容（含 gitignored 目录与仓库外路径），已发生两次实际损失，防御约定如下：

| 已发生事故 | 后果 | 防御措施 |
| --- | --- | --- |
| `.credentials/` 凭证目录被清理 | 推送时才发现 token 失效 | 每次会话开始先做 token 健康检查（`GET /user` 返回 200 才开工）；token 丢失时向使用者索要 |
| `~/.gitconfig` 与 `/home/z/.local/`（git-lfs 二进制）被清理 | 自动快照提交把 156MB gif 按普通 blob 提交，触发 GitHub 100MB 拒收 | LFS filter 同时写入 `~/.gitconfig` 与仓库本地 `.git/config`（后者随 `.git` 持久存活）；git-lfs 二进制丢失需重装至 `/home/z/.local/`（脚本见 worklog Task 10） |
| 2026-10-28 沙箱整体回滚（已跟踪文件与本地 git 历史一并回退至旧快照，`.credentials/`、`/home/z/.local/` 再次丢失） | 本地落后远程 12 commit；全仓库 filemode 伪差异（100644→100755）；推送被 pre-push hook 拦截 | 远程始终保有全部成果：`core.filemode=false` 消除权限伪差异后 `merge --ff-only origin/main` 即恢复；纯文本 commit 可 `push --no-verify` 先行（前提：确认不含 LFS 对象）；git-lfs 3.7.0 已重装、filter 双写已恢复 |

另外约定：沙箱自动快照会在会话间生成 UUID 命名的系统提交，若其内容异常（如绕过 LFS 的大文件），在**未推送**前提下允许 reset 重写本地历史予以修正；已推送历史仍按规则 5 处理。
