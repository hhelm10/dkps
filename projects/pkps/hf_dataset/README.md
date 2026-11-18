---
pretty_name: EEE PKPS Embeddings (EmbeddingGemma-2)
task_categories:
  - feature-extraction
  - other
tags:
  - embeddings
  - model-evaluation
  - benchmark-prediction
  - embeddinggemma-2
  - reproducibility
size_categories:
  - 10K<n<100K
source_datasets:
  - evaleval/EEE_datastore
configs:
  - config_name: eee_69
    default: true
    data_files:
      - split: data
        path: cohorts/eee_69/rows.parquet
  - config_name: eee_45
    data_files:
      - split: data
        path: cohorts/eee_45/rows.parquet
  - config_name: style
    data_files:
      - split: data
        path: cohorts/style/rows.parquet
  - config_name: texts
    data_files:
      - split: data
        path: texts.parquet
  - config_name: targets_69
    data_files:
      - split: data
        path: cohorts/eee_69/targets.parquet
  - config_name: targets_45
    data_files:
      - split: data
        path: cohorts/eee_45/targets.parquet
  - config_name: targets_style
    data_files:
      - split: data
        path: cohorts/style/targets.parquet
  - config_name: style_observations
    data_files:
      - split: data
        path: cohorts/style/observations.parquet
---

# EEE PKPS Embeddings

**Embedding model:** Google's **[EmbeddingGemma-2 (`google/embeddinggemma-2`)](https://huggingface.co/google/embeddinggemma-2)**,
pinned to revision `914f7f89142e33e77833254d9c9b90c3cef7303b`. This model encoded
both queries and responses across all three cohorts, producing L2-normalized
768-dimensional float32 vectors with a 2,048-token input limit.

A private, reusable release of the frozen query and response embeddings used in
the PKPS stress-test follow-ups. It contains **40,327 unique text vectors**, exact
source-text mappings, per-response scores, separate full-depth targets, and the
provenance needed to reconstruct the experiments. The vectors are ready to use
with NumPy; consuming them requires neither the encoder weights nor a GPU.

Release: **1.0.0 · 2026-10-12**. Dataset: `evaleval/eee-pkps-embeddings`.

## Contents

| Cohort | Models | Tasks | Response rows | Full-depth model/task targets | Purpose |
| --- | ---: | ---: | ---: | ---: | --- |
| `eee_69` | 69 | 16 | 30,304 | 947 | Extended, naturally incomplete graph; default for general reuse |
| `eee_45` | 45 | 16 | 23,040 | 720 | Complete five-benchmark cohort; exact original follow-up pool |
| `style` | 44 | 16 | 6,138 | 704 | Original/JSON/blockquote variants of 2,046 selected responses |

The 45-model pool is a subset of the 69-model pool. Style rows comprise three
renderings of each selected response, not 6,138 independent answers. Each of the
three style seeds selects 704 responses (one per model/task); selections overlap
across seeds and are recorded in `cohorts/style/observations.parquet`.

One shared `embeddings.npy` array has shape **(40327, 768)** and dtype **float32**
(about 118 MiB). `q_index` and `r_index` in **every** cohort address this same array.
Exact text shared across cohorts or between query/response roles is stored once.
The first 36,441 vectors come from the extended original-text cache; the remaining
3,886 are additional formatting variants. Packaging preserves every original
embedding value exactly: it performs no new encoding, PCA, or renormalization.

```text
embeddings.npy                       shared float32 vector bank
texts.parquet                        embedding_index -> exact text and SHA256
cohorts/{eee_69,eee_45,style}/
  rows.parquet                       query/response indices and sample scores
  targets.parquet                    full-depth evaluation targets (separate)
  original_indices.parquet           original cache indices -> shared indices
  embedding_config.json              original encoder configuration
cohorts/{eee_69,eee_45}/
  dataset_manifest.json              pinned source runs, hashes, sampling
  run_metadata.json                  evaluation timestamps and configurations
cohorts/style/
  observations.parquet               exact selections for seeds 0, 1, 2
  style_design.json                  eligibility and intervention design
provenance/                          source hashes, encoder, upstream cards
code/                                packaging/preparation/embedding source
MANIFEST.json                        machine-readable inventory and cohort counts
SHA256SUMS                           integrity checks, including the manifest
schemas.json                         machine-readable Parquet schemas
validation.json                      source alignment and vector validation
SCHEMA.md                            field definitions and joins
MANIFESTO.md                         purpose and reuse principles
load_embeddings.py                   standalone reader/verifier/cache exporter
```

## Quick start

Authenticate with a Hugging Face account that can read the private dataset:

```bash
pip install numpy pandas pyarrow huggingface_hub
hf auth login
```

Download a snapshot, record its immutable revision, and load only the desired
row table. For future runs, pass the recorded revision directly to
`snapshot_download` rather than resolving `main` again.

```python
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
import numpy as np
import pandas as pd

repo = "evaleval/eee-pkps-embeddings"
revision = HfApi().dataset_info(repo).sha
print("Dataset revision:", revision)  # Save this with your experiment.
root = Path(snapshot_download(repo, repo_type="dataset", revision=revision))

vectors = np.load(root / "embeddings.npy", mmap_mode="r", allow_pickle=False)
rows = pd.read_parquet(root / "cohorts/eee_69/rows.parquet")
texts = pd.read_parquet(root / "texts.parquet").set_index("embedding_index")

batch = rows.iloc[:32]
query_vectors = vectors[batch.q_index.to_numpy()]       # (32, 768)
response_vectors = vectors[batch.r_index.to_numpy()]  # (32, 768)
response_text = texts.loc[batch.r_index, "text"].tolist()
sample_scores = batch.score.to_numpy()

# Full-depth labels are deliberately an explicit, separate read.
targets = pd.read_parquet(root / "cohorts/eee_69/targets.parquet")
```

The optional standalone helper offers `load_cohort(root, "eee_69")`,
`verify(root)`, and `export_cache(root, destination, "eee_45")`. Its reader never
loads full-depth targets implicitly. After downloading, verify the whole release:

```bash
python /path/to/snapshot/load_embeddings.py /path/to/snapshot --verify
```

For Hugging Face Datasets users, the named configurations expose the Parquet
tables. The split name `data` is a storage label, **not a train/test split**:

```python
from datasets import load_dataset
rows = load_dataset(repo, "eee_69", split="data", revision=revision, token=True)
# Vectors still come from embeddings.npy via snapshot_download + np.load.
```

## Encoder and source provenance

- Encoder: [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2/tree/914f7f89142e33e77833254d9c9b90c3cef7303b),
  revision `914f7f89142e33e77833254d9c9b90c3cef7303b`.
- Data: [evaleval/EEE_datastore](https://huggingface.co/datasets/evaleval/EEE_datastore/tree/8b9632422dd2b3806431efd758c7cdc84044169f),
  revision `8b9632422dd2b3806431efd758c7cdc84044169f`.
- Benchmarks: `math-mc`, `gsm-mc`, `gpqa-diamond`, `judgebench`, `reward-bench-2`.
  Subtasks account for the 16 task IDs.
- Prompt for **both** query and response text: `SentenceSimilarity`, with exact
  prefix `task: sentence similarity | query: `.
- Computation: Apple M5 / MPS, float32, CPU fallback disabled; text encoders only.
  Stored vectors are L2-normalized float32, 768 dimensions, without learned PCA.
- Maximum sequence length: **2,048 tokens**. The text table retains the exact
  input strings before tokenizer truncation; long strings in the regular cohorts
  may have been truncated during encoding. Style response eligibility ensures
  all three renderings fit, including the prompt. This guarantee concerns style
  responses, not all queries.
- Recorded environment: torch 2.14.1, transformers 5.19.0,
  sentence-transformers 6.1.0, NumPy 2.5.3. Reader dependencies are much lighter;
  see `requirements.txt`.

The source selects the latest manifest `added_at` run for each model/benchmark
at the pinned revision. The complete cohort has 225 selected runs; the extended
cohort has 307. Within each model/task, at most 32 eligible scored responses are
sampled deterministically after sorting query IDs, with a model/task SHA256-derived
seed. Full-depth targets are computed **before** this cap. A scored empty response
can contribute to a target while being ineligible for the embedding pool.
Preparation code and selected-run checksums are included.

## Interpretation and experiment design

The release supplies frozen features and observation tables; it does not prescribe
a predictor or establish a query-kernel advantage. It uses EG2 and the corrected
follow-up pipeline, separately from the original paper's Gemini/PCA results.

The extended graph has 947 known cells out of 69 × 16: **157 cells have no observed
target**. Keep those cells absent. `score` is an observed per-response value and
is not always binary; `y_full` is the selected run's full-depth finite-sample mean,
not a population truth. Do not treat the 32-response cap as the original benchmark
depth. Shared texts, correlated model families, and repeated style variants also
mean rows are not independent samples.

Apply observation masks before fitting PCA, choosing bandwidths, or constructing
model-level features. Keep full-depth targets out of cache-only training and
tuning. For full-reference experiments, specify which other developers' labels
are allowed and hold out the complete target developer from label-dependent
selection. Having a frozen vector on disk does not authorize using a hidden
response in the predictor.

Evaluation timestamps refer to selected runs. They are neither publication dates
nor model-release dates, and this snapshot is not a complete version history.
Style interventions preserve exact response content through reversible JSON and
blockquote serialization; they test formatting sensitivity, not every possible
paraphrase or prose style.

The [audit and measured results](https://github.com/evijit/dkps/blob/28249bb/projects/pkps/STRESS_TEST_REVISIONS.md)
describe the original follow-up protocols and their limitations. The dataset's
[manifesto](MANIFESTO.md) states the principles for extending and reusing this cache.

## Reconstructing the original cache layout

`export_cache` restores the original text order, original query/response indices,
and 128-vector `.npy` shards. The 45-model and 69-model targets match their source
tables. Style targets are restricted to the 44 eligible models (704 cells);
the source's extra 16 targets for an unused model remain available in `eee_45`.

```bash
python /path/to/snapshot/load_embeddings.py /path/to/snapshot \
  --export /path/to/new/eee-followups --cohort eee_45
```

The exported directory includes `pool.parquet`, `targets.parquet`, and
`embeddings/{rows.parquet,config.json,complete.json,*.npy}`. Experiment drivers
can consume it through their existing `--source` option. `model_path` and local
raw-data `source` paths are omitted from portable metadata; immutable repository
IDs, revisions, relative source paths, and hashes remain. Set a local model path
explicitly if generating new embeddings. Re-encoding on another backend is not
assumed to reproduce every stored bit.

## Attribution and terms

Credit the EEE datastore contributors, the underlying benchmark sources, Google's
EmbeddingGemma-2 authors, and the PKPS follow-up work when using this release.
The pinned EEE card declares MIT for the collection and retains the terms of
individual original sources. The pinned encoder card declares Apache-2.0.
These upstream declarations are recorded in `provenance/upstream_*_card.md`;
this package does not add a blanket license over third-party benchmark text.
Model weights are not included.

For reproducible attribution, report the dataset ID **and resolved Hub commit**,
cohort name, source and encoder revisions, sampling budget, observation mask,
and permitted reference-label regime.
