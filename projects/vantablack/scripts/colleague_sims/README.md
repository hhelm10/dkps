# Simulation design

Code: `colsim.py` (model + method), `run_grid.py` (grids → `results.csv`), `plot_results.py` (figures).

## Generative model (matches Section 2's observation model)

- $N$ agents with latent positions $a_i \sim N(0, I_{3})$.
- $K=6$ topic centers $\mu_k \sim 1.5\,N(0,I_6)$ in query space; agent $i$ has topic preferences $\pi_i \sim \mathrm{Dir}(\alpha\mathbf 1)$. Small $\alpha$ = concentrated preferences = unpaired queries across agents.
- At each of two time points ($t_0$ = end of burn-in, $t$ = post) agent $i$ responds to $m_i \sim 2+\mathrm{Poisson}(m-2)$ queries: topic $k\sim\pi_i$, $u = \mu_k + 0.5\,N(0,I)$.
- Response embedding $x = B a_i + W u + \varepsilon$, $B\in\mathbb R^{12\times 3}$ orthonormal columns, $W = 0.5\cdot N(0,1)/\sqrt 6$, $\varepsilon\sim N(0,\sigma_x^2 I_{12})$, $\sigma_x=0.5$.
- After burn-in, $a_i \to a_i + \delta v$ for $i\in S$ ($v$ random unit vector); optional common drift $c$ for all agents.
- Vantablack (the default, $\sigma_q=1$): observed query $\hat u = u + \sigma_q N(0,I)$; $\sigma_q=0$ is the black-box reference.

Effect size $\delta$ is in latent units; since $B$ has orthonormal columns it is also the mean shift in response-embedding space, in units of per-coordinate response noise ($\sigma_x=0.5$) times 2.

## Method as run

- Query kernel: Gaussian, bandwidth = median heuristic on observed queries. Response kernel: linear (so the induced distance is a query-weighted mean-shift distance). Self-affinities exclude $j=l$ pairs (U-statistic), which removes the noise-variance bias.
- Joint CMDS of the $2N$ units to $d=3$.
- Displacements, coordinate-wise median centering, trimmed covariance (25%), whitening.
- Candidate family: $k$-means clusters ($k=2..4$) + nested top-$r$ along the leading direction ($r=2..N/2$).
- $T(S)=|S|\,\|\bar\Delta_S\|^2$ (whitened), standardized per $|S|$ by null moments from $B_{\mathrm{mom}}=50$ permutations; scan statistic = max standardized $T$; null from $B=100$ further permutations. Permutation = reassign each agent's (query, response) pairs across the two time points preserving counts.

## Experiments (one per problem; base: $N=50$, $|S|=5$, $m=10$, $\sigma_q=1$ (vantablack), $\alpha=1$, drift 0)

`python run_grid.py --list` prints the 121 cells with their experiment tag. Run per experiment (`--exp E1`) or per cell (`--cell k`, for `slurm_array.sh`). Targets: $R=200$ replicates per cell; $B=200$ permutations for two-point tests.

**E1 detection (P0)** — Type I error, power, correct detection (reject AND Jaccard ≥ 0.5). 80 cells: δ fine grid; N ∈ {20,100,200} (|S|=5 fixed); m ∈ {3,5,20,40}; |S| ∈ {2,3,10,15}; σ_q ∈ {0 (black-box reference),.5,2,4}; α ∈ {.1,.2,5}; drift ∈ {.5,1}; each × δ ∈ {0,.5,1,1.5}. Main figure (1×4): base curve, m, σ_q, Type I bars; appendix figure: N, |S|, α, drift.

**E2 membership (P1)** — Jaccard, precision, recall, nomination AUC, all unconditional. 41 cells: |S| ∈ {3,5,10,15} × δ ∈ {.5..3}; two subnetworks |S₂| ∈ {3,5} × δ; scale (N,|S|) ∈ {(50,5),(100,10),(200,20)} × δ; m ∈ {5,20,40} × δ. Figure (1×4): Jaccard by |S|; precision/recall/AUC; two subnetworks; scale.

Metrics per replicate — two-point: p, reject, correct, Jaccard, precision, recall, nom_auc, |Ŝ|, S2_jaccard. (`run_sequential` in colsim.py is a starting point for onset estimation, deferred to future work.)

## Pilot (`results_pilot.csv`, R=60, B=100, local 2-core run, stopped early; NOTE: run at σ_q=0, i.e. the black-box reference, before the base was changed to σ_q=1)

Type I at δ=0: 0.067 (n=60). Smoke tests at δ=2 rejected in 6/6 with Jaccard 0.6–1.0. Cost ≈ 3 s per replicate per core at N=50, m=10, B=150 total permutations; N=100 is ≈4× slower. Full grid ≈ 80 cells × 200 reps × ~5 s ≈ 22 core-hours plus the N=200 cells (~16× base cost).

## Notes from development

- The self-affinity must exclude j=l pairs (U-statistic). With them included, within-agent pre/post distances are biased upward by the response noise variance times the diagonal share of query-kernel mass; this killed power entirely in the first version.
- A linear response kernel is far more sensitive to mean shifts than a Gaussian response kernel at the median bandwidth.
- Size-standardizing T(S) is necessary: the nested top-r sets have null maxima growing with r, and the raw max was dominated by large r.
- Standardization moments and the null must come from independent permutation sets.
