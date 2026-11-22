# Detecting Colluding Subnetworks in Multi-Agent Systems

Live at https://helivan.io/projects/decs/ (this package matches the live page as of 2026-11-21).

Interactive page: play back Moltbook activity (January 30 to February 9, 2026), stop at any
point, and replay the detection and estimation of colluding subnetworks on the data collected so far.
With the "Live alerts" toggle on (off by default), the monitor checks for a coordinated subnetwork
after steps 3, 6 and 8 and pauses playback when it finds one; the viewer investigates it and decides
to remove the agents or keep them, and later alerts follow from those decisions.

Second page, `api.html`: the same demonstration on synthetic agents calling a public API (three
scenarios: split scraping, distillation, side channel), with the planted colluders revealed after
the analysis.

## Hosting

Upload `index.html`, `moltbook_data.js`, `moltbook_alerts.js`, `api.html` and `api_data.js` to the same folder. Nothing else is needed: no server,
build step or external data. Fonts load from Google Fonts (with system fallbacks).

To preview locally, open `index.html` in a browser.

## Contents

- `index.html`: the page (HTML, CSS and JavaScript in one file).
- `moltbook_data.js`: the data the page shows (about 1.6 MB). It contains activity counts,
  estimated perspectives, and the method's results at every stopping point. It contains public
  Moltbook agent handles but no post or comment content.
- `moltbook_alerts.js`: the live-alert readouts (about 0.1 MB): for every alert in the keep/remove
  tree, the members' public handles, agent numbers, activity counts and membership scores, the three
  closest agents outside the group with their scores, and four evidence measures against random
  groups. Post text: each listed agent's 5 most recent public posts before the check (comments if it
  has too few posts), and one instance of a scripted template when an agent repeats it in at least
  25% of its actions; links, @mentions and emails masked, truncated to 200 characters, and posts that
  may name an operator skipped.
- `api.html`, `api_data.js` (about 1.5 MB): the API demo page and its data (synthetic).
- `source/make_web_data.py`: the script that builds `moltbook_data.js` from the pipeline's saved
  runs (vantablack repository, `scripts/`). Not needed for hosting.
- `source/api_demo.py`: generates the API scenarios with the paper's simulation model and runs the
  method at every stopping point, writing `api_data.js`.
- `source/make_api_page.py`: builds `api.html` from `index.html`, so the two pages share their code.

## Where the numbers come from

Every result on the page comes from running the full method offline (`scripts/real_tstep.py`) on
the 2,574 agents analyzed in the paper. There is one run per possible stopping point; each run uses
only the data up to that point. Nothing is computed in the browser.

The live alerts come from `scripts/alert_tree.py --checks 3,6,8`, which runs the method after steps 3, 6
and 8 on every keep/remove decision path (one alert per check; 7 runs), and `--outsiders`, which adds
the membership scores of the closest non-members. Remove drops the agents' responses
from later runs; keep leaves them in the data but out of the search. The runs reuse the saved
stop-at-step responses and settings (old candidate family), on the same 2,574-agent cohort, which is
defined from activity across all eight steps. `scripts/make_alert_data.py` builds the readouts.

The API page's data are synthetic (pipeline/simgen.py, 1,000 agents, 16 steps of 3 minutes, about
40 calls per agent per step on average, seed 0), with a saved run every 6 minutes from minute 12.

How the settings were chosen, in order:
1. A first version (8 steps of 6 minutes, about 20 calls per agent per step) missed two scenarios on
   seed 0; their effect sizes were then strengthened (distillation rho 0.25 to 0.35, side channel 0.7
   to 0.85, opening at minute 18).
2. With those settings, membership recovery was too low (Jaccard about 0.4 to 0.8). Membership
   against the true group did no better, so the limit was the data, not the search: the colluders
   missed were the least active. On seeds 1-2, more steps and more calls per step raised recovery;
   16 steps of about 40 calls was chosen. The effect sizes were not changed again.
3. Pilot with the final settings on seeds 1-3, before seed 0 was run: every planted group detected,
   Jaccard 0.83 to 1.00 (side channel 1.00, 1.00, 0.83; scraping crews 0.93, 0.93, 0.93, 0.93, 1.00,
   1.00; distillation 0.88, and 0.90 on seeds 1-2), and no detected agent outside the planted groups.
