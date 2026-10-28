# PKPS paper — TODO

Status: the joint response space is the base method everywhere (HELM: standardized
answer text + 92-d shared PCA; EEE: 83-d shared PCA; dimension = what the per-dataset
construction retains in total). All results, figures, ablations, and sensitivity sweeps
are regenerated under this base; the package reproduces every operating point to 0 on
both the new base and the archived published protocols (8/8 validation checks).

## 1. Overleaf (source of truth — apply by hand)

- [ ] Apply the 18 numbered text edits (Table 1 cells, §4 prose, Appendix A protocol
      rewrite, two new ablation paragraphs, updated ablation/sensitivity numbers).
      The full old→new list is in the conversation; the verified numbers behind it are
      archived in the repo history (commit 103b4d8 and its predecessors).
- [ ] Re-upload the 7 regenerated figure PDFs from `projects/pkps/writing/figures/`:
      `fig_suite_query_efficiency`, `fig_qe_conditional`, `fig_suite_completion`,
      `fig_completion_conditional`, `fig_eee_query_efficiency`, `fig_eee_completion`,
      `fig_sensitivity` (no caption changes).
- [ ] After the Overleaf edit, paste the updated `main.tex` back into
      `projects/pkps/writing/` so the local mirror, `main.pdf`, and `main.bbl` can be
      recompiled and recommitted.

## 2. Paper writing still open

- [ ] §5 `[PLACEHOLDER]` Practical-deployment paragraph (a package-oriented draft was
      proposed earlier; needs a decision and Overleaf insertion).
- [ ] ICLR 2027: style-file swap and full figure re-audit when the CFP drops
      (~late Sept deadline).
- [ ] Re-read the abstract/intro for any remaining "block" phrasing once the appendix
      rewrite lands — the main text should read as: one shared space, kernel-gated.

## 3. Ablations — NOT in the paper (user decision 2026-09-16)

The Pipeline Ablations appendix (app:abl) is dropped entirely: the only worthwhile
comparison is to DKPS, handled in the main text. The ablation results stay in the
repo (results-abl/, git-ignored, regenerable from flags) for internal confidence and
potential rebuttals only. Remove all three \ref{app:abl} sites in Overleaf: the
main-text "Ablations of ..." sentence, the Appendix A Ensemble paragraph's closing
sentence, and the app:abl section itself.

## 4. Parked experiments (revisit only if the paper grows an embedding appendix)

- [ ] Embedding-sensitivity study (potion8m, jina v5 family, bge-small, minilm, …) was
      run under the old blocked EEE base (`results-eee-rd1-<tag>/`). Rerun the key
      tags under the joint base before citing any of it in the paper.
- [ ] minilm-chunk variant (chunked mean-pooling to beat the 256-token window):
      embedder script exists, never run to completion.
- [ ] Blocked-era stragglers never swept: jina5-nano-clust, nomic15, qwen06b.
- [ ] Final embedding recommendation table with CPU throughput (potion8m 4776 texts/s
      is the current front-runner; robustness rule: nomic15/gemma300m/jina5-omni
      disqualified, qwen06b/jina5-small too slow on CPU).

## 5. Repo / package housekeeping

- [ ] `results-*` archive sprawl: `-onehot-blocked`, `-blocked`, `-j79`, `-cap240elbow`
      era dirs are all git-ignored local state. Decide what to keep; at minimum keep
      `-onehot-blocked` (HELM published) and `-blocked` (EEE ablation), which
      `validate_package.py --response_space blocked` reads.
- [ ] The EEE `results-eee-rd1-joint`/`-joint-d240` pilot dirs and the HELM
      `anstext-*`/`opttext-joint*` pilot dirs can be deleted once the appendix numbers
      are settled (they are reproducible from flags).
- [ ] README "Tests" line still says "14 tests" — the suite is at 21; refresh when
      next touching the README.
- [ ] Consider a short `docs/` note (or README section) describing the answer-text
      standardization (`data/resolve_answer_option_text.py`) since the MedQA
      query/target columns in the TSVs are known-unreliable — future users of
      `data/med_qa.tsv` should be warned not to trust the stored option ordering.
