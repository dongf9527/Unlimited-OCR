#!/usr/bin/env bash
# 同步上游 baidu/Unlimited-OCR 工作树到主仓库根目录
# - 排除 .git（规则 2：禁止嵌套仓库）、.gitattributes（LFS filter 不适用）
# - LFS 指针文件从 GitHub 媒体端点下载真实文件替换，并按指针中的 sha256 校验
# - .gitignore 采用合并策略（上游 Python 项 + 本仓库安全项），由脚本外部处理
set -euo pipefail

UPSTREAM=/home/z/my-project/research/Unlimited-OCR
REPO=/home/z/my-project
LFS_BASE="https://media.githubusercontent.com/media/baidu/Unlimited-OCR/main"

echo "=== 1. rsync 工作树（排除 .git / .gitattributes / .gitignore）==="
rsync -a \
  --exclude='.git' \
  --exclude='.gitattributes' \
  --exclude='.gitignore' \
  "$UPSTREAM"/ "$REPO"/
echo "rsync 完成"

echo "=== 2. 替换 LFS 指针为真实文件 ==="
cd "$REPO"
for ptr in $(grep -rlE '^version https://git-lfs' --include='*' . 2>/dev/null | grep -v '^\./\.git/' | grep -v '^\./research/' | grep -v '^\./docs/' | grep -v '^\./scripts/'); do
  rel="${ptr#./}"
  # 解析指针中的 oid 与 size
  oid=$(sed -n 's/^oid sha256://p' "$ptr")
  size=$(sed -n 's/^size //p' "$ptr")
  echo "  LFS: $rel (oid=${oid:0:12}... size=$size)"
  curl -sL -o "$ptr" "$LFS_BASE/$rel"
  actual_size=$(wc -c < "$ptr")
  if [ "$actual_size" != "$size" ]; then
    echo "  !! 大小不匹配: 期望 $size 实际 $actual_size"; exit 1
  fi
  actual_oid=$(sha256sum "$ptr" | cut -d' ' -f1)
  if [ "$actual_oid" != "$oid" ]; then
    echo "  !! sha256 不匹配: 期望 $oid 实际 $actual_oid"; exit 1
  fi
  echo "  OK: $rel ($actual_size bytes, sha256 校验通过)"
done

echo "=== 3. 验证 ==="
test ! -e "$REPO/upstream/.git" && test "$(find "$REPO" -name '.git' -not -path "$REPO/.git" -not -path "$REPO/research/*" | wc -l)" = "0" && echo "  无嵌套 .git ✓"
file "$REPO/assets/logo.png" "$REPO/Unlimited-OCR-2.pdf" | sed 's/^/  /'
du -sh "$REPO/assets" "$REPO/wheel" | sed 's/^/  /'
echo "=== 同步完成 ==="
