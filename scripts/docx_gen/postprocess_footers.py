#!/usr/bin/env python3
"""docx 后处理：
1) 删除 document.xml 中的空 <w:pgNumType/>（封面节残留，WPS 兼容）
2) 按节修补页脚 PAGE 域格式开关：第2节(目录)→ROMAN，第3节(正文)→arabic
"""
import re
import shutil
import sys
import zipfile
from pathlib import Path

DOCX = sys.argv[1] if len(sys.argv) > 1 else "/home/z/my-project/download/百度Unlimited-OCR深度拆解报告.docx"
TMP = Path("/home/z/my-project/scripts/docx_gen/_tmp_docx")

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def main():
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    with zipfile.ZipFile(DOCX) as z:
        z.extractall(TMP)

    doc_path = TMP / "word" / "document.xml"
    xml = doc_path.read_text(encoding="utf-8")

    # 1) remove empty pgNumType
    n_removed = len(re.findall(r"<w:pgNumType/>", xml))
    xml = xml.replace("<w:pgNumType/>", "")
    doc_path.write_text(xml, encoding="utf-8")
    print(f"[pgNumType] removed {n_removed} empty tags")

    # 2) map sections -> footer files
    sect_prs = re.findall(r"<w:sectPr[^>]*>.*?</w:sectPr>", xml, re.S)
    print(f"[sections] found {len(sect_prs)} sectPr blocks")

    rels_path = TMP / "word" / "_rels" / "document.xml.rels"
    rels_xml = rels_path.read_text(encoding="utf-8")
    rel_map = dict(re.findall(r'<Relationship[^>]*Id="([^"]+)"[^>]*Target="([^"]+)"', rels_xml))
    # also handle attribute order variance
    for m in re.finditer(r"<Relationship\b[^>]*/>", rels_xml):
        tag = m.group(0)
        rid = re.search(r'Id="([^"]+)"', tag)
        tgt = re.search(r'Target="([^"]+)"', tag)
        if rid and tgt:
            rel_map[rid.group(1)] = tgt.group(1)

    fmts = {}  # footer file -> fmt
    for idx, sp in enumerate(sect_prs, start=1):
        frefs = re.findall(r'<w:footerReference[^>]*w:type="default"[^>]*r:id="([^"]+)"|<w:footerReference[^>]*r:id="([^"]+)"[^>]*w:type="default"', sp)
        rid = None
        for a, b in frefs:
            rid = a or b
        if not rid:
            continue
        target = rel_map.get(rid, "")
        fname = target.split("/")[-1]
        fmt = "ROMAN" if idx == 2 else "arabic"
        fmts[fname] = fmts.get(fname, [])
        fmts[fname] = fmt  # last wins if shared
        print(f"[section {idx}] footer={fname} -> {fmt}")

    for fname, fmt in fmts.items():
        fpath = TMP / "word" / fname
        if not fpath.exists():
            print(f"[warn] {fname} not found, skip")
            continue
        fxml = fpath.read_text(encoding="utf-8")
        patched, n = re.subn(
            r"(<w:instrText[^>]*>)\s*PAGE\s*(</w:instrText>)",
            rf"\1 PAGE \\* {fmt} \\* MERGEFORMAT \2",
            fxml,
        )
        fpath.write_text(patched, encoding="utf-8")
        print(f"[footer] {fname}: {n} PAGE field(s) -> {fmt}")

    # rezip
    out = Path(DOCX)
    tmp_out = out.with_suffix(".tmp.docx")
    with zipfile.ZipFile(tmp_out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(TMP.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(TMP).as_posix())
    tmp_out.replace(out)
    shutil.rmtree(TMP)
    print("[done] repacked", out)


if __name__ == "__main__":
    main()
