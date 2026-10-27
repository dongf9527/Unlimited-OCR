#!/usr/bin/env python3
"""修复 TOC 首条重复：移除 TOC 域段落内 docx-js 残留的 hyperlink 样例条目，
保留脚本注入的 34 条统一占位条目（用户更新域后由 Word 重建真实目录）。"""
import re
import shutil
import sys
import zipfile
from pathlib import Path

DOCX = sys.argv[1]
TMP = Path("/home/z/my-project/scripts/docx_gen/_tmp_fix")

if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir(parents=True)
with zipfile.ZipFile(DOCX) as z:
    z.extractall(TMP)

doc_path = TMP / "word" / "document.xml"
xml = doc_path.read_text(encoding="utf-8")

# 定位 TOC 域 separate 之后、同段落内的 hyperlink（唯一的 hyperlink 元素）
pat = re.compile(
    r'(<w:r><w:fldChar w:fldCharType="separate"/></w:r>)'
    r'<w:hyperlink [^>]*w:anchor="_Toc1">.*?</w:hyperlink>',
    re.S,
)
xml2, n = pat.subn(r"\1", xml)
print(f"removed {n} residual hyperlink entry from TOC field paragraph")

if n:
    doc_path.write_text(xml2, encoding="utf-8")
    out = Path(DOCX)
    tmp_out = out.with_suffix(".tmp.docx")
    with zipfile.ZipFile(tmp_out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(TMP.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(TMP).as_posix())
    tmp_out.replace(out)
    print("[done] repacked", out)
else:
    print("[skip] nothing to fix")
shutil.rmtree(TMP)
