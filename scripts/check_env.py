#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_env.py — PDF-OCR 管线（Unlimited-OCR-2）环境自检脚本

用途：
  在目标机器的目标 Python 环境内运行，逐项核对 docs/prd/pdf-ocr-prd.md §5.1 与
  docs/prd/deployment.md §2/§3 声明的依赖版本、GPU/CUDA、磁盘与模型权重文件，
  输出 PASS/WARN/FAIL 判定与修复提示。纯标准库实现，venv 建到一半也能跑。

用法：
  python scripts/check_env.py                       # 检测当前解释器环境，路线 A+B
  python scripts/check_env.py --route A             # 只按路线 A 判定
  python scripts/check_env.py --model-dir ~/sourcecode/Unlimited-OCR/models/Unlimited-OCR
  python scripts/check_env.py --json                # 机器可读输出（CI 用）
  python scripts/check_env.py --strict              # 有 WARN 也返回非零退出码

退出码：0=无 FAIL；1=存在 FAIL；2=--strict 且存在 WARN

判定基线（pdf-ocr-prd.md §5.1 v1.1 定稿，2026-10-28）：
  Python 3.10（conda 独立环境，禁 3.12+）｜torch 2.11.0+cu128｜transformers 4.52.1（trust_remote_code 依赖）
  tokenizers 0.21.4（随 transformers 联动）｜huggingface-hub 0.36.0｜pymupdf 1.27.2.2
  addict 2.4.0 / einops 0.8.2 / easydict 1.13（模型代码确证 import）
  pillow 11.x｜opencv-python-headless 5.x｜python-docx 1.2.x｜reportlab ≥4.0（定稿区间）
  路线 B 已归档（v1.0 基线：torch 2.13.0 + sglang wheel），--route B 按归档基线校验
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

# ---------------------------------------------------------------- 常量与基线

# torch ↔ torchvision 官方配对表（近期主版本，向下仅列影响面内的）
TORCH_TV_PAIR = {
    "2.9": "0.24", "2.10": "0.25", "2.11": "0.26",
    "2.12": "0.27", "2.13": "0.28",
}

# 文档基线版本（pdf-ocr-prd.md §5.1 v1.1 定稿 / deployment.md v1.1 §2；依据 env-compat-eval.md）
PIN = {
    "torch": "2.11.0",   # v1.1 定稿；路线 B 归档基线 2.13.0，--route B 时切换
    "transformers": "4.52.1",
    "tokenizers": "0.21.4",
    "huggingface_hub": "0.36.0",
    "pymupdf": "1.27.2.2",
    "addict": "2.4.0",
    "einops": "0.8.2",
    "easydict": "1.13",
    "fastokens": "0.3.2",
}
# 实测确认版本的定稿区间（env-compat-eval.md：确认点 pillow 11.0.0 / opencv 5.0.0.93 / python-docx 1.2.0）
SPEC = {
    "pillow": ">=11.0,<12",
    "opencv-python-headless": ">=5.0.0,<6",
    "python-docx": ">=1.2,<2",
    "reportlab": ">=4.0",
}
# import 名 ≠ 发行名的映射；同名可列多个候选发行名
DIST_OF = {
    "pillow": ["pillow"],
    "opencv-python-headless": ["opencv-python-headless", "opencv-python", "opencv-contrib-python-headless"],
    "python-docx": ["python-docx"],
    "pymupdf": ["pymupdf"],
    "reportlab": ["reportlab"],
    "torch": ["torch"],
    "transformers": ["transformers"],
    "tokenizers": ["tokenizers"],
    "huggingface_hub": ["huggingface-hub", "huggingface_hub"],
    "addict": ["addict"],
    "einops": ["einops"],
    "easydict": ["easydict"],
    "sglang": ["sglang"],
    "fastokens": ["fastokens"],
    "flash-attn": ["flash-attn", "flash_attn"],
}
MIN_DRIVER = 570          # PRD §5.1：驱动 ≥ 570（CUDA ≥ 12.8）
MIN_DISK_GB = 40          # deployment.md §1：权重 6.6GB + 环境约 15GB，留余量 40GB
MIN_VRAM_A_GB = 16        # 路线 A 最低
MIN_VRAM_B_GB = 24        # 路线 B 最低
CU_INDEX = "https://download.pytorch.org/whl/cu128"

PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "INFO"
_C = {"g": "\033[92m", "y": "\033[93m", "r": "\033[91m", "b": "\033[96m", "d": "\033[2m", "0": "\033[0m"}
_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def c(tag, s):
    return _C.get(tag, "") + s + _C["0"] if _COLOR else s


def fmt(level):
    return {"PASS": c("g", "✔ PASS"), "WARN": c("y", "⚠ WARN"),
            "FAIL": c("r", "✘ FAIL"), "INFO": c("b", "· INFO")}[level]


# ------------------------------------------------------------ 版本比较工具

def vtuple(v):
    """'2.13.0+cu128'/'0.5.20.dev10' -> 可比较元组（仅数字段，local/后缀段忽略比较）"""
    v = str(v).split("+")[0]
    parts = []
    for seg in v.split("."):
        m = re.match(r"^(\d+)", seg)
        parts.append(int(m.group(1)) if m else 0)
    return tuple(parts + [0] * (4 - len(parts)))


def cmp_ok(installed, op, target):
    """支持 == != >= <= > < ~= 的简化比较"""
    a, b = vtuple(installed), vtuple(target)
    return {"==": a == b, "!=": a != b, ">=": a >= b, "<=": a <= b,
            ">": a > b, "<": a < b,
            "~=": a >= b and a < (vtuple(target)[0] + 1, 0, 0, 0)}[op]


def eval_spec(installed, spec):
    """'>=10.0' / '==4.52.1' / '>=0.21,<0.22' -> 全部子句满足才 True"""
    if not spec:
        return True
    return all(cmp_ok(installed, op, tgt)
               for op, tgt in re.findall(r"(==|!=|>=|<=|>|<|~=)\s*([\w.]+)", spec))


# ---------------------------------------------------------------- 检查框架

class Report:
    def __init__(self):
        self.rows = []          # (level, item, message, hint)

    def add(self, level, item, message, hint=""):
        self.rows.append((level, item, message, hint))

    def has(self, level):
        return any(r[0] == level for r in self.rows)

    def summary(self, route):
        for lv, item, msg, hint in self.rows:
            line = f"{fmt(lv)} [{item}] {msg}"
            if hint:
                line += f"\n     -> {hint}"
            print(line)
        print("-" * 72)
        a_ok = route_a_ok(self)
        b_ok = route_b_ok(self)
        sglang_absent = any(it == "sglang" and "未安装" in m for lv, it, m, _ in self.rows)
        print(f"路线 A（Transformers 单机）: "
              f"{c('g', '可部署') if a_ok else c('r', '存在阻断项')}")
        if sglang_absent and route in ("both", "A"):
            b_line = c("d", "未部署（路线 B 已归档，见 docs/archive/）")
        else:
            b_line = c("g", "可部署") if b_ok else c("r", "存在阻断项")
        print(f"路线 B（SGLang 服务化，归档）:   {b_line}")
        return 0 if not self.has(FAIL) else 1

    def to_json(self):
        return {"findings": [{"level": lv, "item": it, "message": m, "hint": h}
                             for lv, it, m, h in self.rows]}


def dist_version(report, name):
    """取发行版本：优先 importlib.metadata，失败回退包内 __version__"""
    try:
        from importlib import metadata as im
    except ImportError:                                    # py<3.8 不支持，忽略
        im = None
    for d in DIST_OF.get(name, [name]):
        if im is not None:
            try:
                return im.version(d), d
            except Exception:
                pass
        mod = sys.modules.get(name.replace("-", "_"))
        if mod is None:
            try:
                mod = __import__(name.replace("-", "_"))
            except Exception:
                continue
        v = getattr(mod, "__version__", None)
        if v:
            return v, d
    return None, None


def req_spec_for(report, dist, dep):
    """从 transformers 等 dist 的 metadata 里解析其对 dep 的版本要求（如 tokenizers）"""
    try:
        from importlib import metadata as im
        reqs = im.requires(dist) or []
    except Exception:
        return None
    for r in reqs:
        m = re.match(rf"^\s*{re.escape(dep)}\s*([<>=!~].*)", r)
        if m:
            return m.group(1).split(";")[0].strip()
    return None


# ---------------------------------------------------------------- 各项检查

def check_python(rep):
    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) == (3, 10):
        rep.add(PASS, "python", f"Python {ver}（PRD 基准版本 3.10）")
    elif (v.major, v.minor) == (3, 11):
        rep.add(WARN, "python", f"Python {ver}：可用但偏离上游验证基准 3.10",
                "建议按 deployment.md 用 conda/uv 建 python=3.10 独立环境")
    else:
        rep.add(FAIL, "python", f"Python {ver}：PRD 明确禁止 3.12+（3.9 亦不在验证范围）",
                "conda create -n uocr python=3.10 -y 或 uv venv --python 3.10")


def check_gpu(rep, route):
    smi = shutil.which("nvidia-smi")
    driver = None
    if smi:
        try:
            out = subprocess.run([smi, "--query-gpu=driver_version,name,memory.total",
                                  "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=15).stdout.strip()
            if out:
                d0 = out.splitlines()[0].split(",")
                driver = float(d0[0].split("-")[0])
                name, mib = d0[1].strip(), float(d0[2])
                rep.add(INFO, "gpu", f"{name}，{mib / 1024:.1f} GB，驱动 {d0[0]}")
                if driver >= MIN_DRIVER:
                    rep.add(PASS, "driver", f"驱动版本 {d0[0]} ≥ {MIN_DRIVER}（PRD 要求）")
                else:
                    rep.add(WARN, "driver",
                            f"驱动 {d0[0]} < {MIN_DRIVER}，可能不支持 CUDA 12.8 运行时",
                            "升级驱动后重试；老驱动需改用 cu118/cu121 索引并同步调整 torch 版本")
                if route in ("B", "both"):
                    if mib / 1024 >= MIN_VRAM_B_GB:
                        rep.add(PASS, "vram", f"{mib / 1024:.0f} GB ≥ {MIN_VRAM_B_GB} GB（路线 B 门槛）")
                    elif mib / 1024 >= MIN_VRAM_A_GB:
                        rep.add(WARN, "vram",
                                f"{mib / 1024:.0f} GB 介于 {MIN_VRAM_A_GB}-{MIN_VRAM_B_GB} GB：仅路线 A 可跑（单页顺序）",
                                "路线 B（SGLang 生产）建议 A800/A100-80G")
                    else:
                        rep.add(FAIL, "vram",
                                f"{mib / 1024:.0f} GB < {MIN_VRAM_A_GB} GB，两条路线均低于门槛")
            else:
                rep.add(WARN, "gpu", "nvidia-smi 无输出（无 GPU 或驱动未就绪）")
        except Exception as e:
            rep.add(WARN, "gpu", f"nvidia-smi 调用失败：{e}")
    else:
        rep.add(WARN, "gpu", "未找到 nvidia-smi（无 GPU / 驱动未装 / 容器未透传）")

    # torch 侧 CUDA 运行时自洽性
    tv, _ = dist_version(rep, "torch")
    if tv and route in ("A", "both"):
        try:
            import torch
            rep.add(INFO, "cuda-rt", f"torch {tv}（编译 CUDA {torch.version.cuda}）")
            if torch.cuda.is_available():
                n = torch.cuda.device_count()
                caps = [torch.cuda.get_device_capability(i) for i in range(n)]
                rep.add(PASS, "cuda-rt", f"torch.cuda 可用，{n} 卡，算力 {caps}")
            else:
                rep.add(FAIL, "cuda-rt", "torch 已装但 cuda 不可用",
                        "检查驱动/容器透传；或 torch 装成了 CPU 版：pip install "
                        f"torch=={PIN['torch']} --index-url {CU_INDEX}")
        except Exception as e:
            rep.add(WARN, "cuda-rt", f"torch 导入失败：{e}")


def check_torch_stack(rep, route):
    tv, _ = dist_version(rep, "torch")
    if not tv:
        rep.add(FAIL, "torch", f"未安装 torch（基线 {PIN['torch']}+cu128）",
                f"pip install torch=={PIN['torch']} torchvision --index-url {CU_INDEX}")
        return
    pin = vtuple(PIN["torch"])
    if vtuple(tv) == pin:
        rep.add(PASS, "torch", f"torch {tv} == v1.1 定稿基线 {PIN['torch']}")
    elif route == "B":
        rep.add(FAIL, "torch",
                f"torch {tv} ≠ {PIN['torch']}：归档路线 B 的 SGLang wheel 按该基线 torch 编译，版本必须一致",
                f"pip install torch=={PIN['torch']} --index-url {CU_INDEX}")
    else:
        rep.add(WARN, "torch",
                f"torch {tv} ≠ v1.1 定稿基线 {PIN['torch']}：模型代码仅用标准算子，其他版本大概率可跑但未经定稿验证",
                "按 deployment.md §2 重装定稿版本；如需 SGLang（已归档）见 docs/archive/deployment-v1.0.md")
    # torchvision 配对
    tvv, _ = dist_version(rep, "torchvision")
    if tvv:
        want = TORCH_TV_PAIR.get(str(vtuple(tv)[:2][0]) + "." + str(vtuple(tv)[1]))
        if want and vtuple(tvv) == vtuple(want):
            rep.add(PASS, "torchvision", f"torchvision {tvv} 与 torch {tv} 配对正确")
        elif want:
            rep.add(WARN, "torchvision", f"torchvision {tvv} 与 torch {tv} 官方配对应为 {want}",
                    f"pip install torchvision=={want} --index-url {CU_INDEX}")
    else:
        rep.add(WARN, "torchvision", "未安装 torchvision（模型代码 deepencoder 视觉侧需要）",
                f"随 torch 一起安装：--index-url {CU_INDEX}")


def check_transformers(rep, route):
    tf, _ = dist_version(rep, "transformers")
    if not tf:
        rep.add(FAIL, "transformers", f"未安装 transformers（基线 {PIN['transformers']}）",
                f"pip install transformers=={PIN['transformers']}")
        return
    major = vtuple(tf)[0]
    if vtuple(tf) == vtuple(PIN["transformers"]):
        rep.add(PASS, "transformers", f"transformers {tf} == 文档基线（trust_remote_code 依赖此版行为）")
    elif major >= 5:
        if route == "B":
            rep.add(WARN, "transformers",
                    f"transformers {tf} 为 v5 大版本：路线 B 由 SGLang 内核接管推理，但 tokenizer/模板链路仍可能受 v5 重构影响",
                    "路线 B 生产环境建议与文档基线对齐")
        else:
            rep.add(FAIL, "transformers",
                    f"transformers {tf}（v5 大版本）与 trust_remote_code 模型代码不兼容："
                    "Unlimited-OCR 建模源码 import 了 4.x 内部 API（"
                    "transformers.pytorch_utils.ALL_LAYERNORM_LAYERS、"
                    "models.llama.modeling_llama.LlamaAttention/repeat_kv/apply_rotary_pos_emb、"
                    "modeling_attn_mask_utils._prepare_4d_causal_attention_mask、"
                    "utils.import_utils.is_torch_fx_available），v5 已重构/移除上述符号链",
                    f"pip install 'transformers=={PIN['transformers']}' 并联动降级 tokenizers（见下一项）")
    else:
        rep.add(WARN, "transformers",
                f"transformers {tf} ≠ 基线 {PIN['transformers']}：同为 4.x 可试，"
                "但 PRD 标注『必须 4.52.1』（trust_remote_code 行为依赖）",
                f"pip install transformers=={PIN['transformers']}")
    # tokenizers 与 transformers 的联动（读 transformers metadata 里的声明式依赖）
    tk, _ = dist_version(rep, "tokenizers")
    if tk:
        spec = req_spec_for(rep, "transformers", "tokenizers") or f"=={PIN['tokenizers']}"
        if eval_spec(tk, spec):
            rep.add(PASS, "tokenizers", f"tokenizers {tk} 满足 transformers {tf} 的声明要求（{spec}）")
        else:
            rep.add(WARN, "tokenizers",
                    f"tokenizers {tk} 不满足 transformers {tf} 声明要求（{spec}）",
                    f"transformers 与 tokenizers 必须成组安装：transformers=={PIN['transformers']} tokenizers=={PIN['tokenizers']}")
    else:
        rep.add(FAIL, "tokenizers", "未安装 tokenizers（transformers 硬依赖）")


def check_model_side_libs(rep):
    """模型代码确证 import 的三个小库（addict/einops/easydict），缺失即无法加载"""
    for name, why in [("addict", "模型代码 from addict import Dict"),
                      ("einops", "模型代码 from einops import rearrange/repeat"),
                      ("easydict", "模型代码 from easydict import EasyDict")]:
        v, _ = dist_version(rep, name)
        if not v:
            rep.add(FAIL, name, f"未安装 {name}：{why}", f"pip install {name}=={PIN[name] if name in PIN else name}")
        elif name in PIN and vtuple(v) != vtuple(PIN[name]):
            rep.add(WARN, name, f"{name} {v} ≠ 基线 {PIN[name]}（行为稳定，一般无碍）",
                    f"如需对齐：pip install {name}=={PIN[name]}")
        else:
            rep.add(PASS, name, f"{name} {v}")


def check_generic(rep, name, spec, why, fail_if_missing=False):
    v, dist = dist_version(rep, name)
    if not v:
        lvl = FAIL if fail_if_missing else WARN
        rep.add(lvl, name, f"未安装 {name}：{why}", f"pip install '{name}{spec}'")
        return
    if eval_spec(v, spec):
        rep.add(PASS, name, f"{name} {v} 满足定稿区间 {spec}")
    else:
        rep.add(WARN, name, f"{name} {v} 不满足定稿区间 {spec}：{why}",
                f"pip install '{name}{spec}'")


def check_hub_and_weights(rep, model_dir):
    v, _ = dist_version(rep, "huggingface_hub")
    if not v:
        rep.add(WARN, "huggingface-hub", "未安装（权重下载工具链缺失）",
                f"pip install huggingface-hub=={PIN['huggingface_hub']}")
    elif vtuple(v) != vtuple(PIN["huggingface_hub"]):
        rep.add(WARN, "huggingface-hub", f"huggingface-hub {v} ≠ 基线 {PIN['huggingface_hub']}",
                f"pip install huggingface-hub=={PIN['huggingface_hub']}")
    else:
        rep.add(PASS, "huggingface-hub", f"huggingface-hub {v}")
    for cli in ("hf", "huggingface-cli"):
        if shutil.which(cli):
            rep.add(PASS, "hf-cli", f"权重下载命令可用：{cli}")
            break
    else:
        rep.add(WARN, "hf-cli", "未找到 hf / huggingface-cli 命令",
                "pip install -U 'huggingface-hub[cli]' 后用 hf download 下载权重")

    if not model_dir:
        rep.add(INFO, "weights", "未指定 --model-dir，跳过权重文件检查")
        return
    if not os.path.isdir(model_dir):
        rep.add(FAIL, "weights", f"模型目录不存在：{model_dir}",
                "hf download baidu/Unlimited-OCR-2 --local-dir <dir>")
        return
    cfg = os.path.join(model_dir, "config.json")
    if os.path.isfile(cfg):
        try:
            am = json.load(open(cfg, encoding="utf-8")).get("auto_map", {})
            missing = [k for k in ("AutoConfig", "AutoModel") if k not in am]
            if missing:
                rep.add(FAIL, "weights", f"config.json auto_map 缺 {missing}，trust_remote_code 无法生效")
            else:
                rep.add(PASS, "weights", f"config.json auto_map 完整（{', '.join(am.values())}）")
            # remote code 文件与 safetensors
            for entry in am.values():
                mod = entry.split(".")[0]
                if not os.path.isfile(os.path.join(model_dir, mod + ".py")):
                    rep.add(FAIL, "weights", f"缺少远程代码文件 {mod}.py（auto_map 指向 {entry}）")
            shards = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]
            total = sum(os.path.getsize(os.path.join(model_dir, f)) for f in shards)
            if shards and total > 5 * 1024 ** 3:
                rep.add(PASS, "weights", f"safetensors {len(shards)} 分片，共 {total / 1024 ** 3:.1f} GB（预期约 6.6 GB）")
            elif shards:
                rep.add(WARN, "weights", f"safetensors 共 {total / 1024 ** 3:.1f} GB，明显小于预期 6.6 GB（下载不完整？）")
            else:
                rep.add(FAIL, "weights", "目录内无 .safetensors 权重文件",
                        "hf download baidu/Unlimited-OCR-2 --local-dir " + model_dir)
        except Exception as e:
            rep.add(WARN, "weights", f"config.json 解析失败：{e}")
    else:
        rep.add(FAIL, "weights", f"目录存在但缺 config.json：{model_dir}")


def check_disk(rep, model_dir):
    tgt = model_dir if model_dir and os.path.isdir(model_dir) else os.getcwd()
    try:
        free = shutil.disk_usage(tgt).free / 1024 ** 3
        if free >= MIN_DISK_GB:
            rep.add(PASS, "disk", f"{tgt} 可用 {free:.0f} GB ≥ {MIN_DISK_GB} GB")
        else:
            rep.add(WARN, "disk", f"{tgt} 可用 {free:.0f} GB < {MIN_DISK_GB} GB（权重 6.6GB+环境约 15GB）",
                    "清理磁盘或更换 --model-dir / 工作目录")
    except Exception as e:
        rep.add(WARN, "disk", f"磁盘检测失败：{e}")


def check_route_b(rep, route):
    if route not in ("B", "both"):
        return
    absent_lvl = WARN if route == "B" else INFO
    sv, _ = dist_version(rep, "sglang")
    if not sv:
        rep.add(absent_lvl, "sglang",
                "路线 B 未安装（该路线已随 v1.0 归档，Transformers 单机路线不受影响）",
                "恢复流程见 docs/archive/deployment-v1.0.md §3（归档基线 torch 2.13.0）")
        return
    rep.add(PASS, "sglang", f"sglang {sv}（归档路线 B 组件，按 v1.0 基线校验）")
    fv, _ = dist_version(rep, "fastokens")
    if not fv:
        rep.add(WARN, "fastokens", f"未安装 fastokens（--tokenizer-backend fastokens 需要 {PIN['fastokens']}）",
                f"pip install fastokens=={PIN['fastokens']}")
    elif vtuple(fv) != vtuple(PIN["fastokens"]):
        rep.add(WARN, "fastokens", f"fastokens {fv} ≠ 归档基线 {PIN['fastokens']}",
                f"pip install fastokens=={PIN['fastokens']}")
    else:
        rep.add(PASS, "fastokens", f"fastokens {fv}")


def check_flash_attn(rep, route):
    fv, _ = dist_version(rep, "flash-attn")
    if fv:
        rep.add(PASS, "flash-attn",
                f"flash-attn {fv}（已装；本模型 use_mla=false 走 mha_eager，主路径不可达，无实际加速）")
    else:
        rep.add(INFO, "flash-attn",
                "未安装 flash-attn（正常状态，无需安装）：本模型 use_mla=false，解码固定 mha_eager（SR-SWA），"
                "ATTENTION_CLASSES 无 mha_flash_attention_2 键；视觉侧 use_flash_attn=False 源码硬编码，"
                "其 True 分支亦为 torch 原生 SDPA——装与不装对吞吐与结果均无影响",
                "勿向 from_pretrained 传 attn_implementation=\"flash_attention_2\"（KeyError 崩溃）")


# ---------------------------------------------------------------- 路线结论

def route_a_ok(rep):
    blockers = {"python", "torch", "transformers", "tokenizers", "addict",
                "einops", "easydict", "cuda-rt", "vram", "weights"}
    return not any(lv == FAIL and it in blockers for lv, it, _, _ in rep.rows)


def route_b_ok(rep):
    blockers = {"python", "torch", "driver", "vram", "sglang", "fastokens", "weights"}
    return not any(lv == FAIL and it in blockers for lv, it, _, _ in rep.rows)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="PDF-OCR 管线环境自检（基线见 docs/prd/）")
    ap.add_argument("--route", choices=["A", "B", "both"], default="both",
                    help="按哪条部署路线判定（默认 both）")
    ap.add_argument("--model-dir", default=os.environ.get("UOCR_MODEL_DIR"),
                    help="模型权重目录（可选；校验 config/auto_map/safetensors）")
    ap.add_argument("--json", action="store_true", help="输出 JSON（CI 用）")
    ap.add_argument("--strict", action="store_true", help="存在 WARN 也返回退出码 2")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    if args.no_color:
        global _COLOR
        _COLOR = False

    rep = Report()
    if args.model_dir:                      # 支持 ~/ 与 $VAR 写法（含引号包裹时 bash 不展开的场景）
        args.model_dir = os.path.expandvars(os.path.expanduser(args.model_dir))
    if args.route == "B":
        PIN["torch"] = "2.13.0"            # 路线 B 按归档 v1.0 基线校验（SGLang wheel 绑定）
    if not args.json:                       # JSON 模式 stdout 必须是纯 JSON（CI 解析）
        print(f"{c('b', 'PDF-OCR 管线环境自检')}  路线={args.route}  "
              f"解释器={sys.executable}\n基线来源: docs/prd/pdf-ocr-prd.md §5.1 (v1.1) + deployment.md\n" + "-" * 72)

    check_python(rep)
    check_gpu(rep, args.route)
    check_torch_stack(rep, args.route)
    check_transformers(rep, args.route)
    check_model_side_libs(rep)
    check_generic(rep, "pillow", SPEC["pillow"], "图像 IO（v1.1 定稿，实测确认 11.0.0）", fail_if_missing=True)
    check_generic(rep, "opencv-python-headless", SPEC["opencv-python-headless"],
                  "水印抑制与图像裁剪（v1.1 定稿，实测确认 5.0.0.93）", fail_if_missing=True)
    check_generic(rep, "python-docx", SPEC["python-docx"], "DOCX 导出（v1.1 定稿，实测确认 1.2.0）")
    check_generic(rep, "reportlab", SPEC["reportlab"], "文本型 PDF 导出（PRD ≥4.0）")
    check_generic(rep, "pymupdf", f"=={PIN['pymupdf']}", "PDF 渲染（PRD 基线 1.27.2.2）")
    check_hub_and_weights(rep, args.model_dir)
    check_disk(rep, args.model_dir)
    check_route_b(rep, args.route)
    check_flash_attn(rep, args.route)

    if args.json:
        print(json.dumps(rep.to_json(), ensure_ascii=False, indent=2))
        code = 0 if not rep.has(FAIL) else 1
    else:
        print("-" * 72)
        code = rep.summary(args.route)
    if code == 0 and args.strict and rep.has(WARN):
        code = 2
    sys.exit(code)


if __name__ == "__main__":
    main()
