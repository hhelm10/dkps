# Storage schema, version 1.0.0

`schemas.json` records exact Arrow field types and row counts. All integer vector
indices are zero-based. Parquet files do not contain pickled objects. The NumPy
bank is a standard uncompressed `.npy` file suitable for `mmap_mode="r"`.

## Shared vector and text tables

`embeddings.npy`: float32 array, shape `(40327, 768)`, finite unit vectors.
Row `i` corresponds to `texts.parquet.embedding_index == i`.

| Text column | Meaning |
| --- | --- |
| `embedding_index` | Unique position in the shared vector bank |
| `text_sha256` | SHA256 of the exact unprefixed input string encoded as UTF-8 |
| `text` | Exact input string before tokenizer truncation and prompt insertion |
| `is_query` | Text is used as a query in at least one released cohort |
| `is_response` | Text is used as a response in at least one released cohort |

The role flags are not mutually exclusive. Text is deduplicated by exact string,
not query ID or Unicode normalization. A text hash alone is not an embedding
configuration ID: the same text encoded under another configuration is a different
measurement. Bank indices can change in a future release; pin the Hub revision.

## Cohort observations: `cohorts/<cohort>/rows.parquet`

| Column | Meaning |
| --- | --- |
| `row_id` | Zero-based row position within this cohort and release |
| `model_id` | Source model identifier, including developer prefix |
| `task_id` | Suite-prefixed task/subtask identifier |
| `query_id` | Suite-prefixed source sample identifier |
| `suite` | Source benchmark collection |
| `score` | Observed scalar response score; may be fractional |
| `q_index` | Query position in the shared `embeddings.npy` bank |
| `r_index` | Response position in the same bank |
| `variant` | `original`, `json`, or `blockquote`; regular cohorts use `original` |
| `source_run_path` | Relative aggregate companion path in the pinned EEE dataset |
| `evaluation_timestamp` | Selected run's reported evaluation timestamp (UTC) |

The observation key is `(model_id, task_id, query_id, variant)` within a cohort.
`query_id` alone is not a unique observation: different models can answer the
same query, and formatting variants share the original query ID. Do not concatenate
`eee_45` and `eee_69` as independent samples: the former is contained in the latter.

Exact source text is recovered by joining `q_index` or `r_index` to
`texts.parquet.embedding_index`. `source_run_path` joins to `run_metadata.json.path`
in `eee_45` or `eee_69`. Style rows originate from the `eee_45` runs. Sample-file
paths and hashes are in `dataset_manifest.json.runs`; replacing `_samples.jsonl`
with `.json` gives the aggregate companion path.

## Full-depth targets: `cohorts/<cohort>/targets.parquet`

| Column | Meaning |
| --- | --- |
| `model_id`, `task_id`, `suite` | Model/task cell and its source suite |
| `y_full` | Mean over all finite scored rows in the selected source run/task |
| `depth` | Number of scored source rows contributing to `y_full` before the pool cap |

The key is `(model_id, task_id)`. Join it many-to-one to observation rows only
where the experiment permits full-depth labels. The naturally absent 157 cells
have no target row. Style targets are restricted to its 44 eligible models and
are the same values as in the corresponding complete-cohort cells.

## Style selections and original indices

`cohorts/style/observations.parquet` contains `seed`, `model_id`, `task_id`, and
`query_id`: 704 rows per seed, for seeds 0, 1, 2. Join these four fields to the
style rows on the three observation identity fields, then select a `variant`.
This gives one response per model/task for the chosen seed and variant.

JSON responses are `json.dumps({"response": original}, ensure_ascii=False)`;
blockquote responses prefix every line (including empty lines) with `> `.
Round-trip restoration preserves the original string exactly. Source scores are
carried across variants; these are serialization interventions, not newly scored
model generations. The eligibility and excluded-model record is in
`cohorts/style/style_design.json`.

Each cohort's `original_indices.parquet` has `local_index` and `embedding_index`.
It maps every original cache position to the shared bank, including any query or
response text reused across cohorts. `load_embeddings.export_cache` uses it to
reconstruct the original vector order and query/response indices.

## Integrity and provenance

`MANIFEST.json.files` lists each payload file's path, byte count, and SHA256.
It excludes itself and `SHA256SUMS` to avoid circular hashing. `SHA256SUMS` covers
all payload files **and** `MANIFEST.json`; it excludes itself. The Hub commit hash
identifies the entire release, including `SHA256SUMS`.

`provenance/input_file_hashes.json` hashes the input cache files before packaging.
Portable copies of metadata omit machine-local model/raw-data paths; source hashes
therefore refer to the original input files, while manifest hashes refer to the
packaged files. `provenance/encoder.json` records the exact prompt, encoder/data
revisions, and source-code revision; each cohort also retains its original
encoder configuration, including the ordered-text hash and library versions.

`validation.json` records full-cache checks performed during packaging: source
row/text alignment, finite unit float32 vectors, exact vector agreement for reused
texts, and unique UTF-8 text hashes. Packaging fails if an identical text has
different vectors or if encoder settings disagree.
