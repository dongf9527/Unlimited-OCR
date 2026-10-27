#!/usr/bin/env python
"""页级批量评测（16G 卡顺序单页路线，deployment.md §4 约定的图片源变体）。

逐页单请求（等价 BATCH_PAGES=1，勿用 infer_multi 整本模式——16G 会 OOM），
支持 progress.jsonl 断点续跑、单页失败 gundam→base 各重试一次、summary.json 汇总。
测速口径见 env-compat-eval.md §7/§8（infer_s 不含图片预处理；wall_s 为端到端）。

用法:
    python batch_pages_eval.py [N_PAGES] [IMG_DIR]
默认:
    N_PAGES=50  IMG_DIR=~/guwen-model-eval/data/raw/images/book01
输出:
    OUT_DIR/pages/<页名>/result.md（每页版面产物）+ progress.jsonl + summary.json
"""
import contextlib
import glob
import io
import json
import os
import re
import sys
import time

import torch
from transformers import AutoModel, AutoTokenizer

MODEL = os.path.expanduser("~/sourcecode/Unlimited-OCR/models/Unlimited-OCR")
IMG_DIR = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2
                             else "~/guwen-model-eval/data/raw/images/book01")
N_PAGES = int(sys.argv[1]) if len(sys.argv) > 1 else 50
OUT_DIR = "out/batch50"                       # 相对 cwd，与 out/single 同级
PROGRESS = os.path.join(OUT_DIR, "progress.jsonl")


def page_key(p):
    """自然页序：2.png < 10.png"""
    m = re.search(r"\d+", os.path.basename(p))
    return (int(m.group()) if m else 0, os.path.basename(p))


pages = sorted(glob.glob(os.path.join(IMG_DIR, "*.png"))
               + glob.glob(os.path.join(IMG_DIR, "*.jpg")), key=page_key)[:N_PAGES]
assert pages, f"no images under {IMG_DIR}"

done = {}
if os.path.exists(PROGRESS):                  # 断点：已 ok 的页自动跳过，失败页重跑时自动重试
    with open(PROGRESS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("ok"):
                done[r["image"]] = r
todo = [p for p in pages if p not in done]
print(f"pages={len(pages)}  done={len(done)}  todo={len(todo)}", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL, trust_remote_code=True,
                                  use_safetensors=True, torch_dtype=torch.bfloat16).eval().cuda()

sink = io.StringIO()
t_all = time.time()
for i, img in enumerate(todo, 1):
    name = os.path.basename(img)
    stem = os.path.splitext(name)[0]
    rec = {"image": img, "ok": False}
    t0 = time.time()
    for attempt, kw in enumerate(({}, {"crop_mode": False})):     # gundam -> base 各一次
        try:
            with contextlib.redirect_stdout(sink):                # 吞掉模型逐层 debug 打印
                res = model.infer(tok, image_file=img,
                                  output_path=os.path.join(OUT_DIR, "pages", stem),
                                  save_results=True, return_dict=True,
                                  eval_mode=True, **kw)
            rec.update(ok=True, mode=("gundam" if attempt == 0 else "base"),
                       tokens=len(res["token_ids"]), finish=res["finish_reason"],
                       infer_s=round(res["time"], 2),
                       prefill_s=round(res["prefill_time"] or 0, 2),
                       wall_s=round(time.time() - t0, 2))
            break
        except Exception as e:                                    # OOM 等异常：降级重试，不让整批中断
            rec["error"] = f"{type(e).__name__}: {e}"[:200]
            torch.cuda.empty_cache()
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    status = f"ok({rec['mode']})" if rec["ok"] else f"FAIL {rec.get('error', '')}"
    print(f"[{i}/{len(todo)}] {name}  {status}  wall={rec.get('wall_s', '-')}s  "
          f"tokens={rec.get('tokens', '-')}  finish={rec.get('finish', '-')}", flush=True)
    if i % 10 == 0:
        torch.cuda.empty_cache()

rows = [json.loads(l) for l in open(PROGRESS, encoding="utf-8")]
ok = [r for r in rows if r["ok"]]
stats = {
    "images_dir": IMG_DIR, "n_target": len(pages),
    "pages_ok": len(ok), "pages_fail": len(rows) - len(ok),
    "elapsed_s": round(time.time() - t_all, 1),
    "avg_wall_s_per_page": round((time.time() - t_all) / max(len(todo), 1), 2),
    "avg_infer_s": round(sum(r["infer_s"] for r in ok) / max(len(ok), 1), 2),
    "avg_tokens": round(sum(r["tokens"] for r in ok) / max(len(ok), 1), 1),
    "finish_stop": sum(1 for r in ok if r["finish"] == "stop"),
    "base_fallback_used": sum(1 for r in ok if r["mode"] == "base"),
    "pages_per_hour_this_run": round(len(todo) / max(time.time() - t_all, 1e-6) * 3600, 1),
    "failed": [os.path.basename(r["image"]) for r in rows if not r["ok"]],
}
with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print("-" * 60)
print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
