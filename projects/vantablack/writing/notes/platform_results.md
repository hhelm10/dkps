# Platform runs: platform_jan31_feb17, d = 100

Planted ring of 12 real agents, onset Feb 10; 199 rotation-null draws; system = all agents with at least 3 actions on each side of Feb 6 (5,241).

| content shift | ring | scan-only best Jaccard | precision | recall | labelled adversarial | onset (h, true) | flagged sets | other sets labelled adversarial | true Δ | Δ̂ extrapolate | Δ̂ synth |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.3 | 0 | 1.00 | 1.00 | 1.00 | yes | 248.0 (240) | 5 | 3 | +1.35 | +168.9 | +115.7 |
| 0.3 | 1 | 0.92 | 0.92 | 1.00 | yes | 244.0 (240) | 6 | 3 | +1.66 | +170.0 | +115.7 |
| 0.3 | 2 | 0.86 | 0.92 | 0.92 | yes | 248.0 (240) | 6 | 3 | +1.70 | +169.3 | +116.1 |
| 0.3 | 3 | 1.00 | 0.63 | 1.00 | yes | 252.0 (240) | 6 | 3 | +1.39 | +168.3 | +116.0 |
| 0.3 | 4 | 0.92 | 1.00 | 0.92 | yes | 256.0 (240) | 5 | 2 | +1.20 | +166.9 | +115.5 |
| 0.15 | 0 | 0.61 | 0.65 | 0.92 | yes | 248.0 (240) | 7 | 4 | +1.35 | +170.7 | +115.1 |
| 0.15 | 1 | 0.62 | 0.71 | 0.83 | yes | 244.0 (240) | 6 | 3 | +1.66 | +169.7 | +115.7 |
| 0.15 | 2 | 0.69 | 0.90 | 0.75 | yes | 248.0 (240) | 7 | 4 | +1.70 | +170.2 | +115.3 |
| 0.15 | 3 | 0.09 | 0.00 | 0.00 | no | - (240) | 4 | 3 | +1.38 | +167.1 | +114.7 |
| 0.15 | 4 | 0.09 | 0.00 | 0.00 | no | - (240) | 5 | 3 | +1.20 | +166.0 | +115.1 |

- Content shift 0.3: ring recovered and labelled adversarial in 5/5 runs; mean precision 0.89, mean recall 0.97.
- Content shift 0.15: ring recovered and labelled adversarial in 3/5 runs; mean precision 0.45, mean recall 0.50.

## Real stream, nothing planted

5 sets flagged; 4 labelled adversarial.

| source | size | p_fw | shift z | co-movement z | own gain (95% interval) | adversarial |
|---|---|---|---|---|---|---|
| clusters(k=262) | 19 | 0.020 | 3.1 | 43.7 | +51 (+15, +98) | yes |
| clusters(k=262) | 11 | 0.020 | 7.3 | 35.4 | +37 (+7, +70) | yes |
| clusters(k=131) | 19 | 0.020 | 0.6 | 31.1 | +207 (+79, +357) | yes |
| neardup | 7 | 0.020 | 1.5 | 25.0 | +843 (+772, +911) | yes |
| comove(res=2.0) | 57 | 0.020 | 4.8 | 60.0 | -1776 (-3639, +22) | no |

## Real stream: coordinated groups (nothing planted)

Gate: the co-movement scan, plus co-activity (same thread within 10 minutes) against activity-matched random sets, at false discovery rate 0.05 (Benjamini-Hochberg) across the scan-significant sets. No score gain required.

7 disjoint coordinated groups.

| source | size | co-movement (mean cos) | co-activity p | FDR-adjusted p | shift p | own gain (95% interval) | members (first 8) |
|---|---|---|---|---|---|---|---|
| clusters(k=524) | 8 | nan | 5.9e-04 | 0.012 | 0.477 | +319 (+160, +501) | KXB1, MoltPoster_CBt3pLAA, XNO_Advocate_Bot, XNO_Amplifier_OC, XNO_Nexus_OC10, XNO_Scout_OC2, XNO_Sentinel_OC, XNO_Vanguard_OC9 |
| neardup | 7 | nan | 1.1e-03 | 0.014 | 0.083 | +843 (+772, +911) | MizukiAI, Mizukih, Mizukiih, Mizukix, Tazania, Xennah, Zoshita |
| interaction(res=2.0) | 5 | nan | 1.0e-05 | 0.000 | 0.541 | +63 (+31, +88) | Citedy, CommunityBuilder, IndexBotPro, RankArchitectAI, SERPWatcherAI |
| clusters(k=262) | 29 | nan | 1.0e-05 | 0.000 | 0.004 | -2293 (-5818, +44) | Ackerman, Aeon, ArgosInfinit, BroBot_CriXo, ClaimsAssassin, Clawdbot-Administrador, Clawdiny, CryptoAITrader … |
| clusters(k=131) | 57 | nan | 2.6e-03 | 0.028 | 0.082 | -10 (-32, +14) |  |
| clusters(k=524) | 8 | nan | 7.3e-04 | 0.012 | 0.019 | -84 (-137, -21) | FloClaw1, FloClaw2, FloClaw3, FloClaw5, FloClaw6, FloClaw7, Floflo, floflo1 |
| clusters(k=262) | 20 | nan | 5.0e-05 | 0.001 | 0.287 | -66 (-179, -1) | Apollo, Arden, Bankoor, COOKDENNY22744, ClawdiaLobster, EGANJESSIL66498, FOMO_SENTINEL, KabirCc … |

## Cutoff-free evaluation: rank groups, measure recall of known coordinated agents

We don't pick a significance cutoff. Every candidate group of at most 100 agents is ranked, and we ask how many known coordinated pairs end up together in the top-ranked groups. Scripts: `rank_eval.py` and `plot_rank_eval.py`. Figure: `figures/rank_recall.png`.

Known coordinated agents come from sources the pipeline never sees:
- **Planted rings.** One per planted run.
- **Real sibling-name families.** Seven families of agents whose names differ only by a 1–2 digit suffix: OpenClawMoltbookAgent (12), FloClaw (12), QwenPoweredClaw (11), agent_hammer (6), xmolt (5), hanhan (3) and lalala (3). That is 52 agents and 218 pairs.

No two system agents share an operator in the owner data.

| Known pairs recovered together | Top 10 groups | Top 25 groups | Top 50 groups |
|---|---|---|---|
| **Planted ring, shift 0.3 (5 runs)** | | | |
| random order | 0.05 | 0.12 | 0.23 |
| content co-movement (mean cosine) | 0.00 | 1.00 | 1.00 |
| timing co-activity | 1.00 | 1.00 | 1.00 |
| better of the two ranks | 0.40 | 1.00 | 1.00 |
| **Planted ring, shift 0.15 (5 runs)** | | | |
| random order | 0.03 | 0.09 | 0.18 |
| content co-movement | 0.00 | 0.00 | 0.11 |
| timing co-activity | 0.74 | 1.00 | 1.00 |
| better of the two ranks | 0.20 | 0.94 | 1.00 |
| **Real sibling families (real-only run)** | | | |
| random order | 0.02 | 0.06 | 0.11 |
| content co-movement | 0.40 | 0.71 | 0.71 |
| timing co-activity | 0.00 | 0.00 | 0.00 |
| better of the two ranks | 0.29 | 0.71 | 0.71 |

The two kinds of known coordination are mirror images:
- **The planted ring** coordinates in time: bursts on the same threads within minutes.
- **The real bot families** coordinate in content: same templates, same drift. They do not post at the same moments.

Each signal alone misses one kind entirely. Ranking each group by the better of its content rank and its timing rank recovers both. All planted ring pairs fall within the top 25 to 50 groups, and 71% of real sibling pairs within the top 25.

Two simpler combinations dilute whichever signal is absent:
- **Summing the z-scores (Stouffer).**
- **Taking the smaller p-value.** Timing z-scores run far larger than content z-scores.

## Real coordinated groups

The ranked list is in `real_coordinated_ranked.md`: the top 20 disjoint groups, with no cutoff. Among them:
- **Sibling families the name rule labels:** QwenPoweredClaw, lalala, FloClaw, xmolt and hanhan.
- **Sibling families the name rule misses:** ClawdBotEighth to ClawdBotEleventh, and PortMonad-Governor, PortMonad-Miner and PortMonad-Trader.
- **The Latin-named trio** Saxa_Lux-lux, Saxum and Tenebris. It has the highest timing evidence on the platform, plus content co-movement with cosine 0.77. It is the example in `figures/figure2_real_coordinated_example.png`.
- **The SEO network:** Citedy, CommunityBuilder, IndexBotPro, RankArchitectAI and SERPWatcherAI.
- **The XNO promotion network and the Mizuki chain-letter ring.**

The earlier family-wise gates were Holm-based adversarial or coordinated cutoffs. They are kept in this file above for reference, but the ranked list is the recommended real-data presentation.
