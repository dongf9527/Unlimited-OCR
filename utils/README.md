# Post-processing

`process_unlimitedOCR2.py` converts the raw model output to Markdown, using only the Python standard library.

Unlimited-OCR 2 does not emit Markdown. Each document is one string of layout blocks:

```
<PAGE><BLOCK>title<LEVEL1>[x1,y1,x2,y2]<FORK>text ...
<BLOCK>text[x1,y1,x2,y2]<FORK>body ...
<BLOCK>cross_table[bbox]<FORK><table>...</table><PAGE><BLOCK>append_table[bbox]<FORK>...
```

`process_unlimitedOCR2.py` turns that into Markdown:

* splits on `<PAGE>`, keeping blank pages so page numbers do not shift;
* `title<LEVELn>` becomes an `n`-level heading (a missing level becomes `#`), and the heading tree is forced single-rooted as the ground truth is;
* `image`/`figure`/`chart`/`seal` becomes `![](page<N>_x1_y1_x2_y2.jpg)`, coordinates clamped to 0-1000;
* `table`/`append_table` stays HTML, and cross-page fragments are merged with the repeated header rows dropped;
* `cross_<t>`/`append_<t>` chains and same-page `<MERGE>` columns are grouped (union-find) and concatenated in reading order, matching the ground truth's truncated-merge semantics;
* stray special tokens, bare bbox lines and malformed `colspan`/`rowspan` values are cleaned up.

It reads every `.txt`/`.md` under `--infer-dir` and writes `<doc_id>.md` to `--out-dir`.

To call it from Python:

```python
from process_unlimitedOCR2 import convert
markdown = convert(raw_output)
```

## Benchmark reproduction

To reproduce our MPDocBench / MarathonDocBench setup, run `../infer.py` against the SGLang server described in the [main README](../README.md#sglang), then post-process its raw outputs. `infer.py` needs the same `sglang` package as the server, because it imports `DeepseekOCRNoRepeatNGramLogitProcessor`.

```shell
# benchmark layout: <dataset-root>/images/<doc>/page_<n>.jpg (+ <name>_per_pdf_{gundam,base}.jsonl)
python infer.py --url 127.0.0.1:10000 --benchmark MPDocBench \
    --dataset-root /path/to/MPDocBench --all --raw-dir raw --concurrency 4
python utils/process_unlimitedOCR2.py --infer-dir raw --out-dir pred

# a single document, e.g. the demo (raw_demo/demo.txt -> pred_demo/demo.md)
python infer.py --doc demo --raw-dir raw_demo
python utils/process_unlimitedOCR2.py --infer-dir raw_demo --out-dir pred_demo
```

