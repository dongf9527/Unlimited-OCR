#!/usr/bin/env python
"""页级批量评测（16G 卡顺序单页路线，deployment.md §4 约定的图片源变体）。

逐页单请求（等价 BATCH_PAGES=1，勿用 infer_multi 整本模式——16G 会 OOM），
支持 progress.jsonl 断点续跑、单页失败 gundam→base 各重试一次、summary.json 汇总；
页内 [TPS] 心跳（每 100 token 一行，约 9s）+ max_new_tokens 硬上限防 runaway 页拖死整批。
v3：gundam finish=length 且 top1 字符占比 >50%（loop 判据，实测定性见 env-compat-eval.md §9）
    时自动降级 base 重跑救援（产物同路径覆盖，rec 标记 loop/rescued）；
    非 loop 截断（真密排页）保留 gundam 结果进人工复核，不浪费重跑。
测速口径见 env-compat-eval.md §7/§8/§9（infer_s 不含图片预处理；wall_s 为端到端）。

用法:
    python batch_pages_eval.py [N_PAGES] [IMG_DIR]
默认:
    N_PAGES=50  IMG_DIR=~/guwen-model-eval/data/raw/images/book01
输出:
    OUT_DIR/pages/<页名>/result.md（每页版面产物）+ progress.jsonl + summary.json
"""
import collections
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
MAX_NEW_TOKENS = 3000                         # 单页硬上限：3000/11.5 t/s ≈ 4.3 分钟封顶，超限 finish=length
TPS_EVERY = 100                               # 页内心跳：每 100 token 一行 [TPS]（≈9s）


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

class _TpsFilter(io.StringIO):
    """stdout 过滤器：只放行 [TPS] 心跳行，逐层 debug 打印仍吞掉"""
    def write(self, s):
        if "[TPS]" in s:
            sys.__stdout__.write(s)
        return len(s)


sink = _TpsFilter()
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
                                  save_results=True, return_dict=True, eval_mode=True,
                                  max_new_tokens=MAX_NEW_TOKENS, tps_interval=TPS_EVERY, **kw)
            if attempt == 1 and rec.get("loop") and res["finish_reason"] == "stop":
                rec["rescued"] = True                             # v3: loop 页被 base 救回
            rec.update(ok=True, mode=("gundam" if attempt == 0 else "base"),
                       tokens=len(res["token_ids"]), finish=res["finish_reason"],
                       infer_s=round(res["time"], 2),
                       prefill_s=round(res["prefill_time"] or 0, 2),
                       wall_s=round(time.time() - t0, 2))
            if res["finish_reason"] == "stop" or attempt == 1:
                break                                             # stop 即收；base 轮无论 finish 都收尾
            txt = re.sub(r"\s+", "", res["text"] or "")           # v3: gundam length -> loop 判定
            if txt and collections.Counter(txt).most_common(1)[0][1] / len(txt) > 0.5:
                rec["loop"] = True                                # 疑 loop（判据见 §9）-> base 救援，产物被 base 版覆盖
                continue
            break                                                 # 非 loop 截断（真密排）-> 保留，人工复核
        except Exception as e:                                    # OOM 等异常：降级重试，不让整批中断
            rec["error"] = f"{type(e).__name__}: {e}"[:200]
            torch.cuda.empty_cache()
            if rec.get("ok"):
                break                                             # v3: 已有 gundam 结果，base 重试失败不覆盖
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if rec["ok"]:
        status = "ok(" + rec["mode"] + (",rescued" if rec.get("rescued") else "") + ")"
    else:
        status = f"FAIL {rec.get('error', '')}"
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
    "finish_length": sum(1 for r in ok if r["finish"] == "length"),
    "base_fallback_used": sum(1 for r in ok if r["mode"] == "base"),
    "loop_flagged": sum(1 for r in ok if r.get("loop")),          # v3: gundam 判实 loop 的页数
    "length_rescued": sum(1 for r in ok if r.get("rescued")),     # v3: 其中被 base 救回的页数
    "pages_per_hour_this_run": round(len(todo) / max(time.time() - t_all, 1e-6) * 3600, 1),
    "failed": [os.path.basename(r["image"]) for r in rows if not r["ok"]],
}
with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print("-" * 60)
print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
