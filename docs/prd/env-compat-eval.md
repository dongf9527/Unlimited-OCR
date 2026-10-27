# 用户环境依赖版本适配评估

针对使用者提供的实测环境（CNBEJFSOP03 `~/sourcecode/ppt-master`，pip list 摘录，2026-10-27）逐项对照本文档库的版本基线，回答"现有版本能否直接适配 PDF-OCR 管线"。基线来源：`docs/prd/pdf-ocr-prd.md` §5.1 与 `docs/prd/deployment.md` §2/§3；自动化复检工具见 `scripts/check_env.py`（本次评估的全部判定逻辑已固化在该脚本中，可在目标机器一键复核）。

## 1. 逐项对照结论

| 组件 | 文档基线 | 实测版本 | 判定 | 说明 |
| --- | --- | --- | --- | --- |
| torch | 2.13.0 + cu128 | 2.11.0 | 路线 A 可试 ／ 路线 B 不可用 | 模型代码仅用标准算子，路线 A 大概率可跑但偏离验证基线；路线 B 的 SGLang wheel 按 2.13.0 编译，硬性要求一致 |
| torchvision | 与 torch 同源 | 0.26.0 | ✓（配对正确） | torch 2.11 ↔ torchvision 0.26 属官方配对 |
| torchaudio | 管线不使用 | 2.11.0 | 无影响 | |
| transformers | 4.52.1（必须） | 5.19.0 | **✗ 路线 A 阻断** | v5 大版本重构/移除了 trust_remote_code 模型源码依赖的内部 API，详见 §2.1 |
| tokenizers | 0.21.4 | 0.23.2 | 与 transformers 5.19 自洽；不可单独保留 | 0.23.x 恰为 transformers 5.x 的配套；回退 transformers 时必须成组回退 0.21.4 |
| pillow | ≥ 10.0 | 11.0.0 | ✓ | |
| opencv-python-headless | ≥ 4.9 | 5.0.0.93 | ✓（附带条件） | 满足下限但跨 5.x 大版本；管线仅用基础 IO/裁剪 API，建议随冒烟一并确认 |
| python-docx | ≥ 1.1 | 1.2.0 | ✓ | 导出层 |
| pymupdf | 1.27.2.2 | 未在摘录中 | 待检 | PDF 渲染必需，用 check_env.py 确认 |
| einops / addict / easydict | 0.8.2 / 2.4.0 / 1.13 | 未在摘录中 | 待检 | 模型源码确证 import，缺失即无法加载 |
| huggingface-hub | 0.36.0 | 未在摘录中 | 待检 | 权重下载链路 |
| reportlab | ≥ 4.0 | 未在摘录中 | 待检 | 文本型 PDF 导出层 |
| sglang + fastokens | wheel 0.5.20.dev10 / 0.3.2 | 未在摘录中 | 待检 | 仅路线 B 需要 |
| pytorch-crf / torch_c_dlpack_ext / sentence-transformers | 管线不使用 | 0.7.2 / 0.1.5 / 6.0.0 | 无影响 | 但它们的存在表明这是共享科研环境，见 §3 建议 1 |

## 2. 关键判定依据

### 2.1 transformers 5.19.0：路线 A 的硬阻断

判定不是基于"版本号不同"的经验主义，而是基于对仓库内 `hf-model-code/` 建模源码的静态检查：trust_remote_code 机制会把模型目录下的 `modeling_unlimitedocr.py` 及其依赖链（`_sparse_base_impl.py`、`modeling_deepseekv2.py`、`dash_attention.py` 等）在运行时 import 进 transformers 的插件体系，这些文件直接引用了 4.x 的内部 API。具体符号清单如下，全部取自源码 import 语句：

- `transformers.pytorch_utils`：`ALL_LAYERNORM_LAYERS`、`is_torch_greater_or_equal_than_1_13`（版本哨兵常量，v5 清理 torch<2 支持后此类常量属被裁剪对象）；
- `transformers.models.llama.modeling_llama`：`LlamaAttention`、`repeat_kv`、`apply_rotary_pos_emb`、`LlamaRotaryEmbedding`（跨模型私有符号借用，v5 对 llama 建模与注意力接口做过整体重构，函数签名与类结构均已变化）；
- `transformers.modeling_attn_mask_utils`：`_prepare_4d_causal_attention_mask`（私有函数，v5 重写了掩码处理路径）；
- `transformers.utils.import_utils`：`is_torch_fx_available`（torch.fx 追踪集成在 v5 中已移除）。

外部事实与此互相印证：transformers v5.0.0 已于 2026 年 1 月转正（官方发布说明称其为"一次大幅内部清理"），tokenization 后端同步重新设计；HuggingFace 官方仓库 issue #43611 明确记录了 v5.0.0 破坏部分远程代码模型的加载、且 4.x 全系无此问题。PRD 将 transformers 钉死在 4.52.1 并标注"必须，trust_remote_code 依赖此版行为"，针对的正是这一风险。结论：除非将整套 remote code 源码迁移到 v5 API（工作量等同一个小型适配项目，且需上游配合），路线 A 必须回退 4.52.1。

### 2.2 torch 2.11.0：路线 A 可试，路线 B 禁止

路线 A（Transformers 单机）方面，模型源码使用的都是常规算子（Linear、SDPA、einops 重排、bf16 数据类型），没有依赖 2.12/2.13 的新特性，2.11.0 大概率可以跑通；但 2.13.0 是上游验证过的基线，偏离基线意味着 CUDAGraph、自定义核等性能特性不在保证范围，因此判定为"可试但需冒烟确认"而非"可用"。附带的好消息是 torchvision 0.26.0 与 torch 2.11.0 属官方配对，无需调整。

路线 B（SGLang 服务化）方面没有讨论空间：deployment.md §3.1 归档的 SGLang wheel（0.5.20.dev10）是按 torch 2.13.0 编译的，其依赖声明对 torch 为精确 pin，SGLang 的自定义 kernel（sgl-kernel）同样按 torch ABI 编译，torch 2.11.0 会导致 wheel 依赖解析失败或运行期 ABI 错误。若走路线 B，torch 必须按文档重装 2.13.0+cu128。

### 2.3 tokenizers 0.23.2：自洽但必须随 transformers 联动

0.23.2 本身没有问题——它恰好是 transformers 5.x 系列的配套版本，这也侧面说明该环境里 transformers 5.19.0 是完整安装而非残留。问题在于联动关系：transformers 4.52.1 对 tokenizers 的声明要求是 `<0.22`，因此"只回退 transformers、保留 0.23.2"会造成 pip 依赖冲突，两者必须成组回退（transformers==4.52.1 + tokenizers==0.21.4）。同理，该环境中 sentence-transformers 6.0.0 对 transformers 有自己的版本声明，任何降级都会破坏它，这是 §3 建议 1 的直接依据。

## 3. 结论与建议

1. **不要在现有共享环境里降级混装。** CNBEJFSOP03 的环境含 sentence-transformers、pytorch-crf 等其他项目依赖，为管线单独回退 transformers/tokenizers 会破坏这些项目；且 Python 版本未经确认（PRD 要求 3.10、禁 3.12+）。
2. **按 deployment.md 建独立 venv，与现有环境并存。** `conda create -n uocr python=3.10`（或 `uv venv --python 3.10`），在 venv 内按文档装 pinned 版本。现有环境中已验证可用的 pillow 11.0.0 / python-docx 1.2.0 / opencv 5.0.0.93 可直接复用同版本号安装。
3. **若坚持复用 torch 2.11.0 走路线 A**：仅可在不装 SGLang 的纯路线 A venv 中尝试，先跑 deployment.md §2 的单页冒烟，通过后再批量；路线 B 无此选项。
4. **部署前用自检脚本复核**：`python scripts/check_env.py --route A --model-dir <权重目录>`，全绿（无 FAIL）再进冒烟测试；CI 可用 `--json` 模式收集结果。

## 4. 结论采纳状态（2026-10-28）

本文档 §2/§3 的判定已全部采纳进 PRD v1.1（`pdf-ocr-prd.md` §5.1 定稿版本表）与部署文档 v1.1（`deployment.md`）：torch 2.11.0 成为路线 A 定稿基线（torchvision 0.26.0 配对）；transformers 4.52.1 + tokenizers 0.21.4 维持强制 pin 并在部署文档中标注禁升 5.x；pillow 11.0.0 / opencv-python-headless 5.0.0.93 / python-docx 1.2.0 按实测确认版本定版；SGLang 路线 B 因与定稿 torch 栈冲突移出本期，随 v1.0 文档归档至 `docs/archive/`；环境管理定为 conda 独立环境。`scripts/check_env.py` 已同步按 v1.1 基线校验（--route B 时自动切换归档基线 2.13.0）。

## 5. 部署实测回归（2026-10-28，CNBEJFSOP03 uocr 环境）

按 `deployment.md` §2 三步安装全部成功，与 §1 定稿基线逐项一致：torch 2.11.0+cu128 / torchvision 0.26.0+cu128（官方配对）；transformers 4.52.1 + tokenizers 0.21.4 + huggingface-hub 0.36.0 成组落位（torchvision 附带拉入的 pillow 12.3.0 被第二步正确压回 11.0.0，印证"成组安装"设计的必要性）；addict 2.4.0 / einops 0.8.2 / easydict 1.13 / pymupdf 1.27.2.2 / opencv-python-headless 5.0.0.93 / python-docx 1.2.0 全部精确命中；reportlab 5.1.0 满足 ≥4.0。传递依赖 numpy 2.2.6、safetensors 0.8.0 均落在 transformers 4.52.1 声明区间（numpy≥1.17、safetensors≥0.4.3），随冒烟一并观察即可。锁定文件 requirements-lock.txt 与 uocr-lock.yml 已生成，具备异机复现条件。

可选加速步 `pip install flash-attn --no-build-isolation` 在 metadata 阶段失败（`ModuleNotFoundError: No module named 'psutil'`，根因：`--no-build-isolation` 跳过构建依赖后，flash-attn setup.py 的自定义 build_ext 无法 import psutil）。结合对 `hf-model-code/` 的静态检查：config.json `use_mla=false` → 解码固定 `mha_eager`（SR-SWA）；`ATTENTION_CLASSES` 无 `mha_flash_attention_2` 键（上游注释掉）；视觉侧 `use_flash_attn=False` 硬编码且其 True 分支亦为 torch 原生 SDPA——**flash-attn 对本模型主路径不可达，该失败无需修复，跳过即可**。`deployment.md` §2/§6 与 `scripts/check_env.py` 的 flash-attn 判定文案已按此结论修正（原"未装则吞吐下降"表述作废）。

## 6. 冒烟回归（2026-10-28，RTX 5080 / sm_120 / 驱动 621.01）

`check_env.py --route A` 全绿后单页冒烟一次通过：trust_remote_code 加载 + bf16 推理全链路正常，DASH block-sparse 稀疏注意力（Triton JIT，`_POSIX_C_SOURCE redefined` 为无害编译警告）正常激活，S=1519（image tokens 1513，gundam 单页切片），输出结构化版面块（header / page_number / text + bbox + `<FORK>` 内容）与上游 schema 一致，`save_results` 产物正常落盘。至此 v1.1 定稿矩阵在 Blackwell 消费级卡上完成端到端验证。

同轮发现并修正文档契约 bug：`deployment.md` §3 单页示例原按 `(text, n)` 2 元组解包 `infer()` 返回值，实际 `infer()` 默认返回纯字符串（`return_dict=True` 时返回 `text` / `token_ids` / `prompt` 的 dict），仅 `infer_multi()` 固定返回 2 元组（且其为 `infer(crop_mode=False, return_dict=True)` 的兼容包装；`infer()` 本身即单/多页统一入口）。§3 示例已改为 `return_dict=True` 取值并补注返回值契约。

## 7. TPS 打印刷屏定性与测速修正（2026-10-28，源码核对 `modeling_unlimitedocr.py`）

实测 `infer(..., tps_interval=1)` 的定性结论：`tps_interval` 是真实参数（`infer()` 签名内默认 `0`），语义为"每 N 个 token 打印一行"，`1` 即逐 token 刷屏——该建议作废。打印点在 `_uocr_generate()`（L1185-1187）：`if tps_interval and len(gen) % tps_interval == 0` 时打 `[TPS] tokens=N avg=X t/s`（行首自带 `\n`，故每行间还有一个空行）；`0` 为 falsy，**默认即完全静默**。类 `TPSTextStreamer`（L440，带 `recent=` 字段）在 `infer()` 路径未被使用（L1155 构造的是原生 `TextStreamer`，仅负责流式吐字）。

测速正确姿势（免刷屏、且比 `[TPS]` 行更完整）：`_uocr_generate` 返回 dict 自带计时（L1204-1205），`return_dict=True` 时原样透传——`res['time']`（prefill+decode 总耗时）与 `res['prefill_time']`，decode TPS = `len(res['token_ids']) / (res['time'] - res['prefill_time'])`。需要进度采样时用 `tps_interval=50` 量级。另注：`tokens=1` 首行 avg 为失真值（`t_first` 与首个 token 在同一迭代内赋值，L1177-1178，dt≈0.2ms），从 `tokens=2` 起才有参考意义。若连流式正文吐出都不需要，传 `eval_mode=True`（`stream=False`）全程静默，结束后读 `res['text']`。`deployment.md` §3 已同步补测速注记。待补实测维持：100 页批量吞吐（16G 卡 `BATCH_PAGES=1` 顺序）。

## 8. 单页耗时实测（2026-10-28，RTX 5080，冷启动）

- 条件：`book01/2.png` 单页 gundam，S=1519（image tokens 1513），`tps_interval` 默认静默，按 §7 方法读返回 dict 计时字段（修正后的 §3 示例命令原样复跑）。
- 结果：**tokens=211，prefill=6.00s，total=24.37s，decode_TPS=11.5 t/s**（decode ≈18.4s），与首轮逐 token 流式观测（avg 爬升至 ~11.3 t/s）交叉印证一致。
- 解读：① 冷启动 prefill 含 DASH block-sparse Triton kernel JIT 编译（解码前几步 TPS 仅 ~5.5 后爬升到稳态），热稳态复测证实（见下）；② `res['time']` 起于 `_uocr_generate`，不含图片预处理/gundam 切片与模型加载；③ 流式 `TextStreamer` 逐 token decode/print 开销含在 decode TPS 内，基准测试建议 `eval_mode=True` 去除。
- 热稳态复测（同进程双跑，`eval_mode=True` 去 streamer 开销、`save_results=False`）：run1 prefill 2.92s / 10.8 t/s（进程内残留预热）；**run2 即稳态——prefill 0.35s（较冷启动 6.00s 降 94%，证实冷启动开销为 Triton JIT 主导）、decode 11.5 t/s（与带流式打印口径持平 → TextStreamer 开销可忽略，11.5 即真实 decode 速度）、total 18.71s ≈ 190 页/时**。两轮 `finish_reason=stop`，tokens=211 与冷启动逐字符一致（temperature=0 确定性）。
- decode 效率注记：11.5 t/s ≈ 87ms/token，远低于 3.34B bf16 权重流式的显存带宽理论上限——主要受逐 token 框架开销（Python 解码循环 + kernel 启动 + DASH 路由/entmax）影响，HF eager 路线单卡顺序属正常水平；吞吐量级提升依赖批量与服务化路线（A 级卡，deployment.md §1）。
- 定位结论：5080 单请求 ~24s/页（冷）~ 18.7s/页（热稳态），≈ 148~190 页/时，符合 16G 卡"功能验证/小批量"定位（deployment.md §1）；PRD ≥600 页/时/卡验收按既定定位在 A 级卡验证。
- 质量观察（2026-10-28 终验闭环）：211 token 处以自然 EOS 结束；初疑正文止于"…者可"为早停，经 book01 十页跨页接缝终验**解除**（详见 §9 EOS 终验条）——"者可"为页面物理边界断行非漏字；页码块误识"リョ"已定位（2.png page_number 块，疑书名残字切列所致）。

## 9. 页级批量首测（2026-10-28，RTX 5080，book01 全 10 页）

`scripts/batch_pages_eval.py`（§4 快速验证路线，顺序单页 + 断点续跑）跑完 book01 全册（目录实有 10 页，N_PAGES=50 自动取全；分两次运行经 progress.jsonl 续跑合并，后又 done=0 全程单次复跑一遍，两轮 10 页 tokens/finish 逐页一致——temperature=0 确定性再证，连 runaway 页都逐 token 复现）：

- **成功率与模式**：10/10 ok，0 失败（四轮复现，第四轮为 1500 上限轮），0 次 OOM——gundam 模式在 16G 卡全程可用；v3/1500 轮均以 **finish_stop=10/10、finish_length=0** 收官（5.png 计入 base 救援产物），batch 产物层面无截断/垃圾页。
- **正常页口径**（9 页 finish=stop，tokens 87~215，wall 8.05~21.92s 且与 token 数线性正相关；四轮复现逐页 token 数一致——temperature=0 确定性四连证，第四轮 172/211/162/87/175/188/198/215/177 与前三轮逐字相同）：均值 **15.95~16.72s/页 ≈ 215~226 页/时**，与 §8 热稳态 190 页/时（单页 18.7s）口径吻合偏优（样本页较短）；每页 wall 与 infer_s 差合计 <1s，图片预处理开销可忽略。
- **runaway 页封顶行为**（§8 观察点闭环）：5.png 生成 3000 token 撞 `max_new_tokens=3000` 硬上限（finish=length），wall 264.78s（续跑轮）/260.91s（单次轮）≈ 4.3 分钟封顶按设计生效；页内 [TPS] 心跳全程 11.3→12.0 t/s 无衰减——DASH 长解码稳定，属"一直正常解码"而非挂死/劣化。该页 3000 token 为正常页 token 数（87~215）的 14~35 倍。（注：本条为上限 3000 时期的实测记录；脚本 max_new_tokens 已于 2026-10-28 下调至 1500，见下条处置）
- **runaway 页定性：循环退化（repetition loop）实锤，非真密排**（n-gram 检测：去空白 1755 字符中单字"日"循环占 ~80%（top-10-gram "日"×10 出现 1405 次、top-40-gram 出现 1375 次，阈值 ≥3 次判 loop），tail 全为"日"）。形态：5.png 为《欽定四庫全書簡明目錄》目录页，前 9 个 BLOCK（书名/卷次/部类）识别正常，最后一个窄长条 BLOCK（bbox [116,101,149,445]，宽仅 33px 的版心边栏字次列）内容自"欠巳"后陷入"日\n日\n…"单字循环直至上限。
- **归因闭环（base 重跑实验，2026-10-28）**：同页 `crop_mode=False`（base，不切片）→ **tokens=204，finish=stop** 自然收尾，前 9 块与 gundam 版一致，窄列识别出"欠/日/軍/金/書/月/录/目/錄"单字序列（含简体"录"，简繁混排为模型输出风格）——**循环确认为 gundam 窄条切片特有**（切片放大插值致字形糊化 → 单字连锁误判），base 整页缩放可保字形；该页 base 产物可用且仅 ~17s。loop 全扫工具（top-1 字符占比 >50%）对 10 页仅命中 5.png（81%），其余 3~8% 均健康，判据有效。
- **v3 救援逻辑**（`scripts/batch_pages_eval.py` 已升级，离线三分支验证通过）：gundam finish=length 且脚本内自动判定 top1 字符占比 >50% → 降级 base 重跑救援（产物同路径覆盖，rec 标记 loop/rescued，状态行 `ok(base,rescued)`，summary 增 `loop_flagged`/`length_rescued`）；非 loop 截断（真密排页）保留 gundam 结果进人工复核不浪费重跑；gundam 已有结果时 base 重试失败不覆盖。代价：救援页多花一次 base 重跑（~17s/页），吞吐不省但质量救回。**实战验证（第三轮 v3 全程）**：5.png 按设计触发 `ok(base,rescued)`，wall=282.96s（gundam 封顶 ~262s + base ~21s），tokens=204/finish=stop，产物覆盖为 base 版；summary `loop_flagged=1`/`length_rescued=1`，avg_tokens 回落至真实文本量 178.9（此前被 3000 循环垃圾拉偏至 458.5）。
- **吞吐敏感性**：1 页 runaway ≈ 16 个正常页耗时；runaway 率 5% 时整批吞吐降至 ~125 页/时（3000 上限时期口径）。处置：**max_new_tokens=1500** 上限封顶防拖死（2026-10-28 自 3000 下调：loop 页 gundam 产物仅为 base 救援中间态，封顶 4.3→2.2 分钟，救援页整页 wall 283s→~150s 近乎减半；正常页 87~215 token 余量 ~7 倍，真密排页截断阈值同步降至 1500 仍走人工复核通道）+ v3 自动 base 救援（质量口径），不因个别页调大上限、不加 repetition penalty（temperature=0 为评测口径）。1500 上限下同法估算：runaway 页 ~153s（gundam ~132s + base ~21s），runaway 率 5% 时整批吞吐 ~155 页/时，较 3000 上限改善 ~24%。**实测验证（第四轮 v3+1500 fresh 全跑，2026-10-28）**：5.png 救援页 wall=148.72s（gundam 1500 封顶 ~128s [TPS 到 1500 avg=11.5 t/s] + base 救援 ~21s）——预测 ~150s 命中（偏差 <1%），较 3000 上限 282.96s 降 47.4% 近乎减半；book01 含 10% runaway 整批 121.9 页/时（elapsed 295.4s，3000 上限 v3 轮 83.0/433.5s → +46.9%/-31.9%）。
- **summary 口径警示**：断点续跑轮 `pages_per_hour_this_run=61.7` 被续跑分母（6 页）+ runaway 双重拉偏；单次全程轮 89.0（elapsed 404.5s 含加载）、v3 轮 83.0（433.5s 含加载 + 救援多跑一次）、v3+1500 轮 121.9（295.4s 含加载）均为"含 runaway 整批口径"（book01 runaway 率 10%）。四轮正常页均值 15.95~16.72s/页 ≈ **215~226 页/时**（口径稳定），汇报用中值 ~16.3s/页 ≈ 221 页/时。另：done=10/todo=0 纯续跑轮 summary 从 progress.jsonl 重算，统计字段（avg_infer_s/avg_tokens/finish 分布）有效，但 elapsed_s=0.0、pages_per_hour=0.0 属分母保护的正常表现，勿误读。
- **EOS 早停核查闭环（十页跨页接缝终验，2026-10-28）**：谕旨段 1~4 页三处接缝全部拼合（1→2"自應如此辦理‖第此次各省搜訪書籍"、2→3"收藏百種以上者可‖稱為蔵書之家"、3→4"由提要而得‖全書…之欽此"），finish=stop 落点均为页面物理边界，**无早停漏字**；目录页（5~10）条目无断句概念不适用此判法。
- **页码块专项**：10 页中 5 页有 page_number 块（谕旨页无版心码属常见版式），3 正确（5/7/9"二/二/三"）、2 处定点误识——2.png"リョ"（片假名，疑书名残字切列）、8.png"11"（疑"十一"压缩）。辅助块问题不影响正文召回，记已知问题。
- **窄条循环第二案例（4.png，轻度）**：header 窄条（宽 31px）输出 6 个"金"后自然转移至正文块，整页 finish=stop 未触发救援——v3 救援只覆盖 length 级 runaway，轻度循环表现为页眉小列局部噪声，正文无涸；后处理可按块内字符去重过滤。
- **字准核查待办**：形近字误识候选 3 处（3.png"緒閱"疑"繙閱"、"開明書目"疑"簡明書目"，4.png"嘉興"疑"嘉惠"），需对照原图逐字确认——属字准（CER）范畴，与 EOS/结构无关，不阻塞 100 页验收。
- 100 页验收放大：book01 仅 10 页，需指定更大图片目录复用脚本（`python scripts/batch_pages_eval.py 100 <img_dir>`），脚本与断点逻辑不变。第四轮 N=100 实跑已验证参数行为：book01 自动取全 10 页（n_target=10）+ 纯续跑轮秒出 summary；OUT_DIR 固定 `out/batch50`（目录名与页数无关，换大书时建议 rm -r out 或改脚本常量避免新旧进度混染）。
