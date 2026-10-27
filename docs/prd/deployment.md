# 部署文档：PDF-OCR 管线（Unlimited-OCR-2 · Transformers 单机路线）

对应 PRD：`docs/prd/pdf-ocr-prd.md` v1.1 ｜ 更新 2026-10-28 ｜ 命令均在 Ubuntu 22.04 + CUDA ≥ 12.8 验证路径上
v1.0（SGLang 双路线、uv 管理）已归档：`docs/archive/deployment-v1.0.md`

## 1. 硬件与前置条件

| 项 | 要求 |
| --- | --- |
| GPU | ≥ 16GB（16G 卡仅单页顺序；24G 卡 10 页/批；批量生产推荐 A800/A100-80G） |
| 磁盘 | ≥ 40GB（权重 6.6GB + conda 环境约 15GB，另留产物空间） |
| Python | 3.10（conda 独立环境，禁 3.12+） |
| 驱动 | ≥ 570（CUDA ≥ 12.8 运行时） |
| 网络 | 首次需下载权重，之后可离线 |

环境自检（装完环境后运行，无 FAIL 再进冒烟；版本适配分析见 `env-compat-eval.md`）：

```bash
python scripts/check_env.py --route A --model-dir ~/sourcecode/Unlimited-OCR/models/Unlimited-OCR
# CI/批量场景用 --json 取机器可读结果
```

权重下载（一次，离线分发到内网机器）：

```bash
hf download baidu/Unlimited-OCR-2 --local-dir ~/sourcecode/Unlimited-OCR/models/Unlimited-OCR
# 或 huggingface-cli download（hub 0.36.0 均可用）
```

> **路径约定（定稿）**：权重目录统一为 `~/sourcecode/Unlimited-OCR/models/Unlimited-OCR`，全文档四处路径（下载 `--local-dir`、自检 `--model-dir`、冒烟 `name`、批量客户端 `MODEL`）均已按此写死，换机时整体替换即可。Shell 参数可直接写 `~/...`（check_env 自动展开，加引号亦可）；**Python 代码内 `~` 不会自动展开**，必须如本文档用 `os.path.expanduser()`。`models/` 与 `*.safetensors` 已在 `.gitignore` 排除，严防 6.6GB 分片被提交（仓库未对 safetensors 配 LFS）；`data/` 已移出排除，可正常存放需纳管的样本数据，注意体积即可。

## 2. 环境部署（conda）

```bash
conda create -n uocr python=3.10 -y && conda activate uocr

# torch 栈（cu128 源）
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

# 推理栈（transformers 与 tokenizers 必须成组安装，避免解析器漂移）
pip install transformers==4.52.1 tokenizers==0.21.4 huggingface-hub==0.36.0 \
    pillow==11.0.0 addict==2.4.0 einops==0.8.2 easydict==1.13 \
    pymupdf==1.27.2.2 opencv-python-headless==5.0.0.93

# 导出层
pip install python-docx==1.2.0 "reportlab>=4.0"
```

版本依据（详见 `env-compat-eval.md`）：torch 2.11.0 / pillow 11.0.0 / opencv 5.0.0.93 / python-docx 1.2.0 为现网实测确认版本；transformers 4.52.1 + tokenizers 0.21.4 为 trust_remote_code 兼容基线，**禁止升入 transformers 5.x**（remote code 依赖的内部 API 已被 v5 移除）。

环境锁定（部署完成后立即固化，供同构机器复现）：

```bash
pip freeze > requirements-lock.txt
conda env export --no-builds > uocr-lock.yml
# 异机复现：conda env create -f uocr-lock.yml
```

flash-attn：**不适用，无需安装**（基于 `hf-model-code/` 静态检查，2026-10-28 实测确认）。本模型 `use_mla=false`，解码注意力固定走 `mha_eager`（SR-SWA）；`ATTENTION_CLASSES` 中 `mha_flash_attention_2` 已被上游注释，flash-attn 在主路径不可达，**装与不装对吞吐与结果均无影响**；视觉编码器 `use_flash_attn=False` 源码硬编码，即便启用其 True 分支也只调用 torch 内置 SDPA，不依赖 flash_attn 库。安装失败属预期现象，直接跳过该步：

```bash
pip install flash-attn --no-build-isolation
# 实测（2026-10-28）：metadata 阶段报 ModuleNotFoundError: No module named 'psutil'，无需处理，跳过即可
# 若确需安装（如未来上游启用 MLA/FA2 路径）：先 pip install psutil packaging ninja；
#   并备妥 CUDA 12.8 的 nvcc（CUDA_HOME 指向含 bin/nvcc 的完整 toolkit，§2 pip 装的 cuda-toolkit 运行时库不含 nvcc）；
#   再 export MAX_JOBS=4 限内存（每任务约 2GB），源码编译全程约 1~3 小时
```

> **禁止**向 `from_pretrained` 传 `attn_implementation="flash_attention_2"`：`ATTENTION_CLASSES` 无 `mha_flash_attention_2` 键，模型将在加载期以 KeyError 崩溃。

## 3. 冒烟测试

单页（gundam 模式；`return_dict=True` 时返回含 `text` / `token_ids` / `prompt` 的 dict）：

```bash
python - <<'PY'
import os
import torch
from transformers import AutoModel, AutoTokenizer
name = os.path.expanduser("~/sourcecode/Unlimited-OCR/models/Unlimited-OCR")
tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
model = AutoModel.from_pretrained(name, trust_remote_code=True,
        use_safetensors=True, torch_dtype=torch.bfloat16).eval().cuda()
res = model.infer(tok, image_file="page_1.jpg",
                  output_path="out/single", save_results=True, return_dict=True)
print(res["text"][:200], "| tokens:", len(res["token_ids"]))
PY
```

> **返回值契约（实测 2026-10-28 修正）**：`infer()` 默认直接返回**纯字符串**，非 dict 模式不存在 `(text, n)` 元组形态——勿按 2 元组解包（会报 `too many values to unpack`）。`infer()` 本身即单/多页统一入口（`image_file` 传 list 即多页；`crop_mode=None` 自动选路：单页 gundam 切片、多页 base）；`infer_multi(image_files=[...])` 是强制 base 模式的兼容包装，固定返回 `(text, n_tokens)` 2 元组。

> **耗时/TPS 实测（勿用 `tps_interval=1`）**：该参数默认 `0`=完全静默，设为 `N` 表示每 N 个 token 打一行 `[TPS]`，`1` 会逐 token 刷屏（实测定性见 `env-compat-eval.md` §7）；确需进度采样用 `tps_interval=50` 量级。测速直接读 `return_dict=True` 返回 dict 自带的计时字段：`res['time']`（总耗时）与 `res['prefill_time']`，decode TPS = `len(res['token_ids']) / (res['time'] - res['prefill_time'])`；连流式正文吐出也不需要时传 `eval_mode=True` 全程静默。

多页（整本一次请求，prompt 由模型侧自动构造为 `"<PAGE><image>"*n + "Multi page parsing."`；`infer_multi` 固定返回 `(text, n_tokens)` 2 元组）：

```bash
python - <<'PY'
import os
import torch
from transformers import AutoModel, AutoTokenizer
name = os.path.expanduser("~/sourcecode/Unlimited-OCR/models/Unlimited-OCR")
tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
model = AutoModel.from_pretrained(name, trust_remote_code=True,
        use_safetensors=True, torch_dtype=torch.bfloat16).eval().cuda()
pages = ["page_1.jpg", "page_2.jpg", "page_3.jpg"]   # 按页序排列
text, n = model.infer_multi(tok, image_files=pages,
                            output_path="out/multi", save_results=True)
print(text[:200], "| tokens:", n)
PY
```

## 4. 批量客户端（PDF → 图 → 推理 → 产物）

生产批处理骨架（渲染、分批、断点判断均可直接复用进管线 M2）：

```python
import json, os, tempfile
from pathlib import Path
import fitz
from transformers import AutoModel, AutoTokenizer
import torch

MODEL = os.path.expanduser("~/sourcecode/Unlimited-OCR/models/Unlimited-OCR")
BATCH_PAGES = 10            # 显存受限降批：10 -> 5 -> 1
DPI = 300                   # OOM 降档：300 -> 200

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL, trust_remote_code=True,
        use_safetensors=True, torch_dtype=torch.bfloat16).eval().cuda()

def pdf_to_images(pdf, dpi=DPI):
    """整本渲染为 PNG，返回文件路径列表（页序即列表序）"""
    doc, mat, paths = fitz.open(pdf), fitz.Matrix(dpi/72, dpi/72), []
    with tempfile.TemporaryDirectory() as d:
        for i, page in enumerate(doc):
            p = f"{d}/p{i:04d}.png"
            page.get_pixmap(matrix=mat).save(p)
            paths.append(p)
    doc.close()
    return paths

def parse_pdf(pdf, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    texts, n_total = [], 0
    pages = pdf_to_images(pdf)
    for i in range(0, len(pages), BATCH_PAGES):          # 分批整本请求
        batch = pages[i:i + BATCH_PAGES]
        text, n = model.infer_multi(tok, image_files=batch,
                                    output_path=str(out / f"batch_{i:04d}"),
                                    save_results=True)
        texts.append(text); n_total += n
    (out / "full.md").write_text("\n\n".join(texts), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps(
        {"pages": len(pages), "tokens": n_total, "model": "Unlimited-OCR-2"},
        ensure_ascii=False), encoding="utf-8")

parse_pdf("demo.pdf", "out/demo")
```

断点续跑：每文档完成后写 `manifest.json`（含 doc_id 与产物哈希），批处理入口启动时先查 manifest，已存在则跳过该文档；单页失败重试 2 次（gundam→base 各一次）后落 `failed_pages.json` 继续，行为约定见 PRD §4.6。

**快速验证（50 页量级 / 图片源）**：`scripts/batch_pages_eval.py` 按**顺序单页**逐页请求（等价 `BATCH_PAGES=1`；勿改用 `infer_multi` 整本模式，16G 必 OOM），实现 `progress.jsonl` 断点续跑（已 ok 页自动跳过、失败页重跑自动重试）、gundam→base 重试、`summary.json` 汇总（成功率/均耗时/均 token/finish 分布）。页内可见性：`tps_interval=100` 心跳（约 9s/行，debug 打印仍被过滤）+ `max_new_tokens=3000` 单页硬上限（≈4.3 分钟封顶）——防个别密页/runaway 页让整批"看似假死"。**v3 loop 自动救援**：gundam 超限且脚本内判定 top1 字符占比 >50%（循环退化判据，实测定性见 `env-compat-eval.md` §9——book01 目录页窄条切片单字循环）时自动降级 base 重跑覆盖产物（状态行 `ok(base,rescued)`，summary 增 `loop_flagged`/`length_rescued`）；真密排截断页保留 gundam 结果进人工复核。用法 `python scripts/batch_pages_eval.py [N页数] [图片目录]`，默认 50 页、`~/guwen-model-eval/data/raw/images/book01`。**RTX 5080 实测（2026-10-28，book01 全 10 页，三轮复跑逐页一致，v3 轮 finish_stop=10/10 收官）**：10/10 成功、0 OOM；正常页（finish=stop）16.0~16.7s/页 ≈ 215~226 页/时，1 页 runaway 被 3000 上限封顶后由 v3 自动 base 救援（`ok(base,rescued)`，wall=282.96s，tokens 3000 垃圾→204 可用）——runaway 率 5% 时整批吞吐降至 ~125 页/时，口径详见 `env-compat-eval.md` §9。

## 5. 导出层依赖与后处理

```bash
python utils/process_unlimitedOCR2.py --help   # 后处理：页切分/标题树/跨页表格合并（纯标准库）
```

导出五格式（Markdown/TXT/DOCX/JSON/文本型 PDF）所需依赖已随 §2 装毕（python-docx、reportlab、opencv-python-headless、pymupdf）；导出层实现与产物 schema 见 PRD §4.5，DOCX/文本型 PDF 的中文字体注册与一致性校验在 M3 交付。

## 6. 显存与性能调优速查

| 症状 | 处置 |
| --- | --- |
| 整本请求 OOM | 降批页数 `BATCH_PAGES` 10→5→1；仍 OOM 降 DPI 300→200 |
| 16G 卡 | 固定单页顺序（`BATCH_PAGES=1`），bf16 保持不变，不要降精度 |
| 吞吐低于目标 | 确认输入为 300 dpi 单页图而非整页大图重复缩放；排查 PDF 渲染（fitz）与推理是否串行阻塞；flash-attn 与吞吐无关（§2，主路径不可达） |
| 输出被截断 | 提高模型侧页 token 上限后重跑该页；频繁截断的超密页拆分后再解析 |
| 跨页表格合并变差 | 检查分批边界是否切在同一段表格中间；批大小 ≥ 10 页/批可覆盖绝大多数跨页表格 |
| 加载即报 transformers 内部 API 缺失 | 环境混入 transformers 5.x，回退 `transformers==4.52.1 tokenizers==0.21.4` 并重建环境（勿在共享环境修补） |

> **RTX 5080 实测基线（2026-10-28，单页 gundam，S=1519 / 211 token）**：冷启动 prefill 6.00s（含 DASH Triton JIT）/ 总 24.4s ≈ 148 页/时；**热稳态（同进程二次调用）prefill 0.35s / decode 11.5 t/s / 总 18.7s ≈ 190 页/时**，`finish_reason=stop` 自然 EOS。**页级批量首测（book01 全 10 页）**：正常页 16.3s/页 ≈ 221 页/时（与热稳态口径吻合），runaway 页被 `max_new_tokens=3000` 封顶 264.8s 且 TPS 全程 11.3~11.5 t/s 无衰减。16G 卡定位功能验证/小批量，≥600 页/时/卡验收在 A 级卡。测速方法与口径见 `env-compat-eval.md` §7/§8/§9。

## 7. 验收清单

- [ ] `python scripts/check_env.py --route A --model-dir <权重目录>` 无 FAIL
- [ ] 单页冒烟：Markdown 输出含标题与表格 HTML
- [ ] 100 页 PDF 分批请求成功率 100%，页切分数与输入页数一致
- [ ] 1,000 页连续批处理达标：≥ 600 页/小时/卡，页级失败率 ≤ 0.5%
- [ ] 断点续跑：kill 后重启，已完成文档跳过
- [ ] 五格式产物抽查内容一致，`manifest.json` 哈希登记完整

## 8. 与 v1.0 的差异

| 项 | v1.0（归档） | v1.1（本版） |
| --- | --- | --- |
| 推理路线 | A（Transformers）+ B（SGLang 服务化，生产默认） | 仅 Transformers 单机（B 归档） |
| torch | 2.13.0+cu128（随 SGLang wheel 绑定） | 2.11.0+cu128（现网实测定稿） |
| 环境管理 | uv venv / conda 双写 | 统一 conda |
| pillow / opencv / python-docx | ≥ 10.0 / ≥ 4.9 / ≥ 1.1 | 11.0.0 / 5.0.0.93 / 1.2.0（实测定稿） |
| 延迟/吞吐目标 | 2.5 s/页、1,000 页/小时/卡 | 5 s/页、600 页/小时/卡（单请求推理特征重定） |
