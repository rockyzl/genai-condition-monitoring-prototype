# Pipeline Run Manifest

- Run id: `run_b`  ·  stages executed: **1** (skipped: **1**)
- Seed: **42**  ·  dataset: **FD001**  ·  RUL cap: **125**  ·  git: `694b40e`
- Journal: `/tmp/pytest-of-lu2/pytest-24/test_run_stage_then_skip_secon0/journal.jsonl` (append-only NDJSON, one line per step event)

## Stage DAG

```mermaid
flowchart TD
  s01_ingest["s01 ingest"]
  s02_eda["s02 eda · skipped-not-selected"]
  s03_preprocess["s03 preprocess · skipped-not-selected"]
  s04_features["s04 features · skipped-not-selected"]
  s05_model["s05 model · skipped-not-selected"]
  s06_select["s06 select · skipped-not-selected"]
  s07_predict["s07 predict · skipped-not-selected"]
  s08_evidence["s08 evidence · skipped-not-selected"]
  s09_diagnose["s09 diagnose · skipped-not-selected"]
  s10_eval["s10 eval · skipped-not-selected"]
  s01_ingest --> s02_eda
  s01_ingest --> s03_preprocess
  s03_preprocess --> s04_features
  s04_features --> s05_model
  s05_model --> s06_select
  s05_model --> s07_predict
  s07_predict --> s08_evidence
  s08_evidence --> s09_diagnose
  s09_diagnose --> s10_eval
```

## Stage cards

### s01_ingest — ⏭ skipped (cached)

**What.** Load the raw C-MAPSS FD001 train/test text files into named columns and compute the capped training RUL target.

**Why.** Everything downstream refers to named sensors and a well-defined target; parsing the fixed-width files once here prevents silent column drift.

**功能.** 读入原始 C-MAPSS FD001 训练/测试数据，给每列起名，并算出封顶后的剩余寿命标签。

**目的.** 后面所有环节都按传感器名字来引用，先在这里一次性把固定宽度的文本解析好，避免列错位却没人发现。

- Observed: rows **n/a**, outputs **159 B**, time **0.000s**
- Inputs: `data/raw/CMAPSSData/train_FD001.txt`, `data/raw/CMAPSSData/test_FD001.txt`, `data/raw/CMAPSSData/RUL_FD001.txt`
- Outputs: `data/processed/ingest_manifest.json`
- Assumptions:
  - FD001 only: one operating condition, one fault mode (HPC degradation).
  - Training RUL is capped at 125 cycles — a documented modelling choice, not a property of the data.
