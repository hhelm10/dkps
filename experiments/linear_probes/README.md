# Portable linear probes for trace embeddings

This directory is a standalone experiment: copy it to another project without
the rest of `dkps`, QUENCH, API credentials, or an embedding model installation.
It requires Python 3.11+ with NumPy and SciPy. `export_repo.py` is the **only** file that knows
about this repository's cache layout. The runner does not import it.

The question is: **which attributes can a supervised linear readout recover
from raw, common-rubric, and query-specific-rubric embeddings?** Higher probe
accuracy means greater recoverability; it does not always mean a better
representation. These are new measurements, not rescaled versions of the old
radar spokes.

## Run on the existing caches

From the repository root (or substitute your own Python environment):

```sh
.pixi/envs/default/bin/python experiments/linear_probes/export_repo.py --out data/linear_probes
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .pixi/envs/default/bin/python experiments/linear_probes/run.py --manifest data/linear_probes/manifest.json --out experiments/linear_probes/results
.pixi/envs/default/bin/python -m unittest discover -s experiments/linear_probes -p 'test_*.py'
```

The exporter refuses to overwrite an existing export. The runner checkpoints
after every representation/target pair and resumes when its inputs, source
hashes, and configuration match. Use a different output prefix for a changed
experiment. No API calls are made and no embeddings are regenerated.

Outputs are `results.md` (table), `results.csv`, and `results.json` (predictions,
class-specific recall, class counts, selected penalties, all inner/outer group
assignments, input/source hashes, and environment versions). The CSV also
includes ordinary accuracy and a training-majority-class baseline.

For handoff, `metadata.csv` and `repo_manifest.json` in this directory are
snapshots of the exported labels and representation/target definitions. The
large NPZ files remain in ignored `data/linear_probes/`, not alongside the
code. Copy those files next to the manifest to reproduce the cached run, or
have your colleague export their own embeddings against these row IDs and
update the manifest. `findings.md` summarizes the completed experiments;
`results_seed29.*` records a separate split-seed sensitivity check.

## Bring your own embeddings

Supply three kinds of file, in one directory:

1. `metadata.csv`: one row per trace, with a unique string `row_id`, target
   columns, and grouping columns. Empty labels or groups are excluded separately
   for each target, identically for every representation.
2. One NPZ per representation, containing `row_id` (Unicode array, not Python
   objects) and `X` (finite embeddings of shape `N × D` or `N × sections × D`).
   Row order may differ: the runner joins by ID and rejects duplicate, missing,
   or extra IDs. Export the common intersection if coverage differs.
3. A `manifest.json` describing the files and prediction tasks:

```json
{
  "metadata": "metadata.csv",
  "representations": {
    "my_embedder_raw": "raw.npz",
    "my_embedder_generic": "generic.npz",
    "my_embedder_qubric": "qubric.npz"
  },
  "targets": {
    "harness": {"label": "harness", "group": "task"},
    "task": {"label": "task", "group": "system"}
  }
}
```

For example, metadata can begin as follows (a real experiment needs enough
groups and examples of every class):

```csv
row_id,system,task,harness
system_a/task_1,system_a,task_1,harness_a
system_b/task_1,system_b,task_1,harness_b
system_a/task_2,system_a,task_2,harness_a
```

Save embeddings with `np.savez_compressed(path, X=embeddings,
row_id=np.asarray(ids, dtype=str))`, then run:

```sh
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 python run.py --manifest /path/to/manifest.json --out /path/to/results
```

Optional flags: `--representations name ...`, `--targets name ...`,
`--alphas 0.00001 0.0001 0.001 0.01 0.1 1`, `--outer-folds 5`,
`--inner-folds 3`, `--seed 17`, and `--bootstrap-draws 1000`.
Add any categorical behavioral annotation as another target without changing
the runner. There are no hardcoded class names or embedding dimensions.

The saved sensitivity checks use the same command as the primary run, with
`--seed 29 --out experiments/linear_probes/results_seed29`, and separately
`--representations openai_raw nomic_raw --targets system --alphas 1e-7 1e-6
1e-5 1e-4 1e-3 1e-2 0.1 1 --out experiments/linear_probes/results_ridge_sensitivity`.
Run these sequentially to avoid dense-linear-algebra resource contention.

## Classifier and preprocessing

Each section is L2-normalized independently, then sections are concatenated
with equal weight and the concatenation is scaled by `1/sqrt(sections)`.
A two-dimensional input is treated as one section. Zero blocks are rejected.
No PCA, per-task centering, or across-example normalization is applied. The
classifier has a training-fitted intercept. In particular, held-out task
identity is **not** used to construct its features.

The readout is a class-balanced, multiclass **linear ridge classifier**, not
logistic regression or a nonlinear classifier. It fits one-hot labels using

```
minimize over W,b:
    (1/n) sum_i w_i ||x_i W + b - onehot(y_i)||² + alpha ||W||²
    w_i = n / (K × training_count_of_class(y_i))
predict: argmax_k (x W + b)_k
```

The intercept is unpenalized. Class weights are computed only on the current
training fold. An eigendecomposition of the weighted, training-centered
**linear** Gram matrix solves the objective globally. This dual form is
equivalent to an ordinary linear classifier on the normalized input vectors;
it avoids a large feature-space inverse for six-section embeddings. Scores
are not calibrated probabilities. A unit test compares its predictions to
the direct feature-space solution over multiple regularization strengths.

This implementation is intended for panels of a few thousand traces. Its
dense Gram matrix uses quadratic memory and its eigendecomposition has cubic
time complexity in training-row count; use a different solver before scaling
to tens of thousands of traces. The input/export format need not change.

Regularization is selected from six strengths by pooled balanced accuracy in
three grouped inner folds, then refitted on the outer training set and tested
on its untouched outer fold. Five outer folds provide one prediction per row.
Exact ties favor stronger regularization. The same seed, rows, and group
assignments are used for all representations. A training fold missing any
target class raises an error; the runner never silently drops difficult classes.

## What is held out?

The repository export defines these tasks:

| Target | Label source | Held-out unit | Meaning |
|---|---|---|---|
| System | Exact submission ID | Task | Recognize an existing configuration on new tasks |
| Model | Reported `model_display` | Task | Recover the reported model on new tasks |
| Vendor | Reported `model_org` | Task | Recover a coarse lineage/vendor label on new tasks |
| Harness | Reported `agent` | Task | Recover the reported agent implementation on new tasks |
| Task | Benchmark instance ID | Reported model group | Recognize a known task across systems with held-out model tags |
| Outcome | Resolved/not resolved | Reported model group | Predict correctness across systems with held-out model tags |

Model, vendor, and harness probes see the same system configurations during
training and testing, but on **disjoint tasks**. Thus, model/harness performance
can partly reflect recognition of configuration-specific signatures. This is
a test of recoverable identity information, **not** proof that model and
harness effects have been separately identified. A stronger cross-configuration
test requires multiple independent configurations per class and an appropriate
grouped split; many current labels have only one configuration.

For task/outcome, all systems sharing the same disclosed model tag stay
together. Unknown or mixed-model systems are singleton groups. Reported names
are not canonical parameter IDs: aliases and undisclosed models can still
share underlying parameters across folds. No claim of complete parameter-level
independence is made. Colleagues can substitute curated group labels in the CSV.

## Metrics and interpretation

The primary metric is **balanced accuracy** (mean per-class recall), because
the classes have very unequal frequencies. Its no-information population
baseline is `1/K`, including for a constant classifier. This is not the old
radar's nearest-neighbor chance calculation. Ordinary accuracy, training-
majority accuracy, and per-class recall are also retained.

Intervals bootstrap the held-out **groups**, not individual traces, using
1,000 resamples of the fixed out-of-fold predictions. They describe variation
across the observed task/model groups; they do not include retraining
uncertainty, new judges, or new benchmarks. The default run uses one seeded
nested partition. Group membership is fully recorded for auditing.

Important boundaries:

- Accuracy measures linear accessibility after the stated sample-local
  normalization, not all mutual information. Failure of this classifier does
  not establish invariance to nonlinear readouts.
- Lower identity recovery is not automatically beneficial: identity is
  correlated with real behavior and ability. Relate it to outcome/QUENCH
  performance before making a mechanism claim.
- Correctness is not a behavioral annotation. The available labels do not
  directly measure evidence gathering, localization, editing strategy, or
  verification. There is deliberately no synthetic “Behavior” score here.
- Common and query-specific rubrics have six embedding blocks; raw uses two
  (head and tail). Ridge tuning and equal total norm help control scale, but
  do not make the representations equal-dimensional. Dimensions are recorded.
- Task-conditioned rubrics can reveal task identity through the rubric itself.
  Task recovery is not, on its own, evidence that trace behavior was preserved.
- This experiment isolates the cached descriptions/embeddings **before**
  QUENCH's consensus-centering step. It does not test the complete centered
  QUENCH representation, and is not numerically comparable to the old radars.

## Cache and label provenance

The export pairs the three constructions on the same traces for two embedders:
OpenAI `text-embedding-3-small` and `nomic-ai/nomic-embed-text-v1.5`, using
`gpt-5.4-mini` rubric extractions. Raw is cached head + tail, not the whole trace;
the Nomic raw cache used the first/last 32,000 characters of rendered text.
The OpenAI raw cache uses the first/last 8,000 `cl100k_base` tokens, after
pretrimming to the first/last 40,000 characters (`leaderboard_baselines.py`).
Compare constructions within each embedder; their raw truncation protocols
are not identical across embedders.

OpenAI matrix caches carry system/task IDs. Legacy Nomic caches do not: their
orders are reconstructed from the generating scripts, checked against the
labeled OpenAI panel, and cross-checked by matching the full Nomic qubric
matrix to the companion raw-cache qubric matrix. The exported files always
carry explicit IDs, so this historical assumption is confined to the adapter.

Wholly empty, missing, or unparseable common/qubric descriptions are excluded
from **all** representations. The manifest lists excluded traces. Individual
blank rubric sections remain as their cached text embeddings. Unknown, mixed,
and clearly non-model `model_display` tags are omitted from the model target;
vendor and harness use their available metadata independently. Labels are
reported tags, not a curated ontology: aliases and version-specific names are
retained. Inspect or revise `metadata.csv` before interpreting fine-grained
comparisons. Vendor is a coarse proxy, not the old radar's keyword-based family
definition.
