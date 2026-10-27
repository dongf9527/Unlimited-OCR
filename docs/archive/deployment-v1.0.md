# 部署文档：PDF-OCR 管线（Unlimited-OCR-2）

对应 PRD：`docs/prd/pdf-ocr-prd.md` v1.0 ｜ 更新 2026-10-27 ｜ 命令均在 Ubuntu 22.04 + CUDA ≥ 12.8 验证路径上

## 1. 硬件与前置条件

| 项 | 路线 A（Transformers） | 路线 B（SGLang，生产默认） |
| --- | --- | --- |
| GPU | ≥ 16GB（RTX 5080 可跑单页顺序） | ≥ 24GB，推荐 A800/A100-80G |
| 磁盘 | ≥ 40GB（权重 6.6GB + 环境约 15GB） | 同左，另留产物空间 |
| Python | 3.10（两条路线各自的独立 venv） | 3.10 |
| 网络 | 首次需下载权重，之后可离线 | 同左 |

环境自检（目标 venv 内运行，逐项核对版本/GPU/磁盘/权重，全绿再进安装与冒烟）：

```bash
python scripts/check_env.py --route A --model-dir /data/models/Unlimited-OCR-2
# CI/批量场景用 --json 取机器可读结果；版本适配分析见 env-compat-eval.md
```

权重下载（一次，离线分发到内网机器）：

```bash
hf download baidu/Unlimited-OCR-2 --local-dir /data/models/Unlimited-OCR-2
# 或 huggingface-cli download（hub 0.36.0 均可用）
```

## 2. 路线 A：Transformers 单机部署

```bash
conda create -n uocr python=3.10 -y && conda activate uocr
pip install torch==2.13.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.52.1 huggingface-hub==0.36.0 tokenizers==0.21.4 \
    pillow addict==2.4.0 einops==0.8.2 easydict==1.13 \
    pymupdf==1.27.2.2 opencv-python-headless
```

冒烟测试（单页）：

```bash
python - <<'PY'
import torch
from transformers import AutoModel, AutoTokenizer
name = "/data/models/Unlimited-OCR-2"
tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
model = AutoModel.from_pretrained(name, trust_remote_code=True,
        use_safetensors=True, torch_dtype=torch.bfloat16).eval().cuda()
text, n = model.infer(tok, image_file="page_1.jpg",
                      output_path="out/single", save_results=True)
print(text[:200], "| tokens:", n)
PY
```

多页（整本一次请求）：`model.infer_multi(tokenizer, image_files=[...], output_path="out/multi", save_results=True)`。

## 3. 路线 B：SGLang 服务化部署（生产）

### 3.1 环境（uv 管理，Python 3.10）

```bash
uv venv --python 3.10 && source .venv/bin/activate
uv pip install wheel/sglang-0.5.20.dev10+g8d62f5fb9c-py3-none-any.whl   # 仓库 wheel/ 已归档
uv pip install pymupdf==1.27.2.2 fastokens==0.3.2
uv pip install torch==2.13.0 torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 3.2 启动服务（内存参数已按上游文档调优）

```bash
export SGLANG_UOCR_EANABLE_DOC_LIMIT=1        # 页窗调度开关（变量名 EANABLE 为上游拼写，原样保留）
export SGLANG_UOCR_DOC_GUNDAM_PAGE_WINDOW=10  # 每窗页数（gundam 模式）
export SGLANG_UOCR_DOC_BASE_PAGE_WINDOW=10
export SGLANG_UOCR_MAX_PAGE_TOKENS=20000      # 页级输出上限，必须 > 0
export SGLANG_OPT_UOCR_SAM_REL_POS_BIAS_MB=64 # 注意力偏置分块，512 默认会触发 ~462MiB 单次分配
export SGLANG_UOCR_SHM_WRAP_MODEL_DATA=1      # 大张量走 /dev/shm，避免 pickle 复制
export SGLANG_UOCR_SHM_WRAP_MIN_MB=1
export SGLANG_VLM_CACHE_SIZE_MB=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m sglang.launch_server \
  --model-path /data/models/Unlimited-OCR-2 \
  --served-model-name Unlimited-OCR-2 \
  --chat-template unlimited-ocr \
  --tokenizer-backend fastokens \
  --tokenizer-worker-num 8 \
  --mem-fraction-static 0.7 \
  --max-running-requests 64 \
  --max-prefill-tokens 2048 \
  --enable-custom-logit-processor \
  --enable-streaming-session \
  --allow-auto-truncate \
  --disable-radix-cache \
  --mm-preprocess-cache-size-mb 0 \
  --cuda-graph-backend-decode full --cuda-graph-max-bs-decode 64 \
  --cuda-graph-backend-prefill full \
  --cuda-graph-max-bs-prefill 16384 \
  --cuda-graph-bs-prefill 1024 2048 \
  --cuda-graph-config '{"prefill": {"backend": "full", "full_prefill_max_req": 64}}' \
  --host 0.0.0.0 --port 10000
```

健康检查：

```bash
curl -s http://127.0.0.1:10000/health && echo SERVICE_OK
```

### 3.3 客户端调用（OpenAI 兼容）

Python（PDF→图→请求→逐页文本，生产客户端骨架）：

```python
import base64, json, tempfile, requests, fitz
from sglang.srt.sampling.custom_logit_processor import DeepseekOCRNoRepeatNGramLogitProcessor

URL = "http://127.0.0.1:10000/v1/chat/completions"
NGRAM = DeepseekOCRNoRepeatNGramLogitProcessor.to_str()

def pdf_to_images(pdf, dpi=300):
    doc = fitz.open(pdf); mat = fitz.Matrix(dpi/72, dpi/72); paths = []
    with tempfile.TemporaryDirectory() as d:
        for i, page in enumerate(doc):
            p = f"{d}/p{i:04d}.png"; page.get_pixmap(matrix=mat).save(p); paths.append(p)
        blobs = [open(p, "rb").read() for p in paths]
    doc.close(); return blobs

def encode(b):
    return {"type": "image_url",
            "image_url": {"url": "data:image/png;base64," + base64.b64encode(b).decode()}}

def parse_pdf(pdf):
    blobs = pdf_to_images(pdf)
    prompt = "<PAGE><image>" * len(blobs) + "Multi page parsing."
    body = {
        "model": "Unlimited-OCR-2",
        "messages": [{"role": "user",
                      "content": [encode(b) for b in blobs] + [{"type": "text", "text": prompt}]}],
        "max_tokens": 8000 * len(blobs), "temperature": 0,
        "skip_special_tokens": False,
        "images_config": {"crop_mode": len(blobs) == 1},
        "custom_logit_processor": NGRAM,
        "custom_params": {"ngram_size": 35,
                          "window_size": 1024 if len(blobs) > 1 else 128,
                          "page_token_id": 128827, "eos_token_id": 1,
                          "max_page_tokens": 20000},
        "stream": False,
    }
    r = requests.post(URL, json=body, timeout=6000); r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

print(parse_pdf("demo.pdf"))
```

curl 最小验证（单张图）：

```bash
IMG=$(base64 -w0 page_1.jpg)
curl -s http://127.0.0.1:10000/v1/chat/completions -H 'Content-Type: application/json' -d "{
  \"model\": \"Unlimited-OCR-2\",
  \"messages\": [{\"role\": \"user\", \"content\": [
    {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/jpeg;base64,$IMG\"}},
    {\"type\": \"text\", \"text\": \"<image>Parse the document.\"}]}],
  \"max_tokens\": 8000, \"temperature\": 0, \"stream\": false}" | jq -r '.choices[0].message.content'
```

### 3.4 导出层依赖（管线机器与服务机可分离）

```bash
pip install python-docx reportlab opencv-python-headless pymupdf==1.27.2.2
python utils/process_unlimitedOCR2.py --help   # 后处理：页切分/标题树/跨页表格合并（纯标准库，无需额外依赖）
```

## 4. 参数调优速查

| 症状 | 处置 |
| --- | --- |
| 高并发 OOM | `--mem-fraction-static` 0.7 → 0.65；`--max-prefill-tokens` 2048 → 4096 会升峰值，方向反了就降 |
| 超大 batch 回退 eager（变慢） | `--cuda-graph-bs-prefill` 最大桶必须 ≥ `--max-prefill-tokens` |
| 16/32 并发回退 eager | `--cuda-graph-config` 的 `full_prefill_max_req` 默认 8 太小，设为 64 |
| 单页输出被截断 | 提高 `SGLANG_UOCR_MAX_PAGE_TOKENS`（须同步提高 `max_tokens` 预算） |
| 跨页表格合并变差 | 页窗 `SGLANG_UOCR_DOC_GUNDAM_PAGE_WINDOW` 调回 10 或增大 |
| 启动即 OOM（偏置分配） | `SGLANG_OPT_UOCR_SAM_REL_POS_BIAS_MB=64`（等价且峰值更低） |

## 5. 验收清单

- [ ] `/health` 200，服务常驻
- [ ] 单页冒烟：Markdown 输出含标题与表格 HTML
- [ ] 100 页 PDF 一次请求成功率 100%，页切分数与输入页数一致
- [ ] 32 并发压测 30 分钟无 OOM、无 eager 回退（`nvidia-smi` 峰值 < 80%）
- [ ] 断点续跑：kill 后重启，已完成文档跳过
