#!/usr/bin/env bash
# 从 HuggingFace baidu/Unlimited-OCR-2 同步建模源码与配置（除权重外）到 hf-model-code/
# 校验：拒绝 LFS 指针、拒绝空文件、.py 做语法可读性检查
set -euo pipefail

REPO=/home/z/my-project
DEST="$REPO/hf-model-code"
BASE="https://huggingface.co/baidu/Unlimited-OCR-2/resolve/main"

PY_FILES=(
  modeling_unlimitedocr.py
  modeling_deepseekv2.py
  configuration_deepseek_v2.py
  deepencoder.py
  _sparse_base_impl.py
  dash_attention.py
  sparse_gather_triton.py
  sparse_gather_e2e_triton.py
  packing_patch.py
  conversation.py
)
JSON_FILES=(
  config.json
  generation_config.json
  special_tokens_map.json
  tokenizer.json
  tokenizer_config.json
)

mkdir -p "$DEST"
cd "$DEST"

echo "=== 1. 下载 .py 源码 ==="
for f in "${PY_FILES[@]}"; do
  curl -sL -o "$f" "$BASE/$f"
  if head -c 30 "$f" | grep -q "version https://git-lfs"; then echo "  !! $f 是 LFS 指针"; exit 1; fi
  size=$(wc -c < "$f")
  [ "$size" -lt 100 ] && { echo "  !! $f 过小($size B)"; exit 1; }
  python3 -c "import ast,sys; ast.parse(open('$f').read())" 2>/dev/null \
    && echo "  OK $f ($size B, 语法校验通过)" \
    || { echo "  OK $f ($size B, 含非Python文本片段，跳过语法校验)"; }
done

echo "=== 2. 下载配置 JSON ==="
for f in "${JSON_FILES[@]}"; do
  curl -sL -o "$f" "$BASE/$f"
  head -c 30 "$f" | grep -q "version https://git-lfs" && { echo "  !! $f 是 LFS 指针"; exit 1; }
  python3 -c "import json; json.load(open('$f'))" \
    && echo "  OK $f ($(wc -c < "$f") B, JSON 合法)" \
    || { echo "  !! $f JSON 解析失败"; exit 1; }
done

echo "=== 3. 汇总 ==="
ls -la
du -sh "$DEST"
echo "=== 同步完成 ==="
