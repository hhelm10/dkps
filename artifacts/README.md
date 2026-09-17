# Paper artifacts

The canonical, current version of everything that appears in the paper
("Embedding Agentic Traces for Efficient Evaluation"). Regenerate with
`python scripts/collect_artifacts.py` after re-rendering any figure —
sources live in `figures/` and `notes/`; nothing here is hand-edited.

| Artifact | File(s) | Source script |
|---|---|---|
| 1. Hero table (MAE, 6 methods x regime x benchmark x m) | `table1_hero.{png,md,tex}` | `scripts/hero_table.py` |
| 2. Radars (linear-probe metric, 7 embedders) | `fig1_radar.png` | `scripts/fig_radar_probes.py` |
| 3. PKPS geometry (raw vs qubric x n,m) | `fig2_pkps.png` | `scripts/fig2_nm.py` |
| 4. Protocol-robustness MAE grid | `fig3_protocols.png` | `scripts/fig_protocols.py` |
| 5. Ablations under fixed protocol (n refs / adaptive / embedder) | `fig4_ablations.png` | `scripts/fig_ablations.py` |
| 6. Cost-to-achieve (MAE + 1v1 ranking) | `fig5_cost.png` | `scripts/fig_cost.py` |
| 7. Rubric-size sensitivity | `fig6_sensitivity.png` | `scripts/rubric_sens_paper.py` |

Protocol everywhere: leave-one-family-out, consensus centering, per-draw
pooled CV, +/-1 SEM. Style canon: `scripts/hv_style.py`.
