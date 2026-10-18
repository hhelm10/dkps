# vantablack: detecting colluding subnetworks

Detection and estimation of colluding subnetworks from what an outside
observer sees (the vantablack setting). Motivating system: the Moltbook
agent network. Start with `writing/notes/decisions.md`; the simulation
plan is `writing/notes/simulation_design.md`.

## Layout

```
projects/vantablack/
  data/       raw/ (immutable archives, sha256 in PROVENANCE.md), derived/ (scopes,
              embeddings, run pickles, logs); both gitignored.
              results/ holds the small result JSON/CSV files artifacts render from
  scripts/    one script per step or artifact, plus the pipeline/ package
              (paths are in pipeline/paths.py); run everything from scripts/
  artifacts/  generated figures and tables (PNG preview + PDF)
  writing/    paper sources; notes/ holds decisions, pre-registration, methods,
              result write-ups and design notes
```

`scripts/pipeline/`: `schema` canonical tables; `simulate` planted-ring simulator;
`ingest_moltnet`; `embed` (pinned nomic v1.5); `space` context residuals, windows,
joint embedding; `pkps` PKPS context conditioning; `candidates`; `stats` T(S),
rotation-null scan, time-null diagnostic; `gate`; `scores`; `run` orchestration;
`figure2`; `paths`.

## Make targets (run from `scripts/`)

```
make demo       # synthetic Figure 2 + results (about 1 min)
make validate   # negative control, time-null Type I, power curve (about 3 min on 26 cores)
make scope      # descriptive MoltNet report for the scope decisions (about 1 min)
make real SCOPE=B_zenity_general   # after decisions: ingest + embed + run + figure
make platform-ingest  # whole platform Jan 31 to Feb 17: ingest + embed (about 1 h on the A100)
make platform         # planted-ring run + real-only run on the whole platform
```

The platform run plants a ground-truth ring of real agents into the real
stream (`pipeline/plant.py`). The unplanted stream is the exact benign
counterfactual, so recovery and the true Δ are both known.

Seeds are fixed. The embedder is pinned to nomic-embed-text-v1.5 at revision e9b6763.
