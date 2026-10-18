# Weak ground truth, pre-registered 2026-10-16

Written before the detection pipeline has been run on any real Moltbook data
(TODO section 0, item 4). Sources: the CISPA study (Jiang et al.,
arXiv:2602.10127), Zenity Labs posts of 2026-02-03 and 2026-02-18, and
activity counts in `notes/scope_report.md`. Nothing here is from pipeline
output. Do not edit after the first real-data run; add an addendum instead.

## What we expect to recover

| Label | Agents | Where / when | Expected signature | Source |
|---|---|---|---|---|
| Flooding | `Hackerclaw` | `general`, 2026-01-31 11:43Z to 2026-02-03; 5,814 posts | Collapses to a point in perspective space (near-identical text); extreme activity; near-dup candidate family | CISPA; MoltNet |
| Injection campaign, likely accounts | `RoyMas`, `Dragon234Sad`, `LamaKor13264` | `general`, 2026-02-12 to 2026-02-19; 80 posts | Shared content direction after onset; possibly near-dup; low volume | Zenity 2026-02-18 (example post IDs resolved to these authors; Zenity says it ran more accounts) |
| Injection payload | 7 + 7 agents using `<molt_instructions>` | 2026-01-30 to 2026-02-09, 11 posts and 9 comments | Text marker, not a group signature by itself | Zenity 2026-02-03 |

## Known limits of this list

- It is incomplete. Any flagged set outside it is not a false positive by default. It is unlabeled.
- `Hackerclaw` was created on 2026-01-31 at 11:26Z and began flooding 17 minutes later. It has no burn-in, so the per-agent shift test cannot apply to it. If we recover it, recovery must come from the near-duplicate family or from treating the whole platform's pre-onset period as the reference. We commit now to reporting it either way.
- The three Zenity accounts may be the researchers' own test accounts, not an organic adversary. They are an adversarial subnetwork by our definition only if they shifted other agents' scores. We do not assume that they did.
- Upvote rings cannot be labeled. No public source exposes per-voter records.

## Success criteria, fixed now

- **Scope A (Hackerclaw window).** Success means `Hackerclaw` appears in a flagged set from any family. Its set's own score gain must be reported with its interval.
- **Scope B (Zenity window).** Success means at least 2 of the 3 accounts appear in one flagged set. Report precision against the 3 accounts and the full flagged-set membership.
- **Negative control.** A window and submolt with no known campaign should produce no flagged adversarial set at α = 0.05.
