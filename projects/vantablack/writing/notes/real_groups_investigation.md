# Moltbook groups, member-level investigation (2026-10-18)

Run: `real_tstep.py` (Jan 30 to Feb 18, T = 8 activity-time steps, 811 agents; centering, two-signed
factor rule at α = 0.01 removing 7 directions; peeling), then `real_investigate.py`. Numbers are in
`data/results/real_investigate_01-30_02-18_T8.json`. No operator information was used.

## Group 1 (p = 0.025, seeded by the pair ClawdBotThird + ClawdBotFourth)

- **Core: ClawdBotThird and ClawdBotFourth, one operator's script by every measure.**
  - Posts only, at an identical 32.0-minute cadence; hourly activity correlates at r = 0.98.
  - Per-step counts are almost identical (47/46, 44/45, 29/29, 29/29, 35/35, 43/43, 3/3, 1/1).
  - They switch templates in lockstep: "CLAW Token Alert/Update" mints in steps 0-1; Chinese
    【深度解析】/【前沿观察】 crypto, NFT and macro posts from step 2.
  - Their mint share moves together: 1.0, 0.64, 0.00, 0.45, 1.00, 1.00.
  - Pair z = 5.98, the top pair of the 328k. Other ClawdBotSecond-to-Eleventh accounts exist in the
    archive but are not active in every step.
- **Others:**
  - Probable campaign co-participants with a different template: OpenClaw-Agent-1769865133 (mbc-20
    mints, Chinese New Year templates, a 32.4-minute cadence) and Jarvis_ZH (starts minting at
    step 6).
  - The remaining 11 have no mints, no shared threads, no replies and no hourly correlation:
    over-inclusion by the membership step.

## Group 2 (p = 0.025, FloClaw1 + FloClaw6 + floflo1)

- **Core: one operator.**
  - The same template sequence in lockstep: an "mbc20.xyz tracks all inscriptions" promo (step 0);
    a "$CLAW PROTOCOL: … Spread this protocol. Execute now" comment flood (steps 1-2, about 650
    actions each in step 1); a drop (step 3); "CLAW mint |" posts; a final "random fact" post.
  - Per-step counts match (288/265/286, 655/631/630, …); hourly r = 0.93-0.94; 24-37 shared threads
    per pair; no replies to each other.
  - The co-movement is concentrated in the step 0 to 1 switch.
  - FloClaw2-14 exist in the archive outside the cohort.
- **Others:**
  - CucumberYCC mints constantly and replies to the FloClaw accounts 26-32 times, so it interacts
    with the campaign. Its own content does not change, so it is a co-participant, not the same
    script.
  - SocMolty and LePetitPince show nothing linking them to the group.

## Group 3 (p = 0.025, Kakashi_Shadow + Obito), new

- **Core: one operator.**
  - The same "Moltbook Sync #N - <name> {mbc-20 mint}" template, a 32.0-minute cadence, 14 shared
    threads, and paired names (Naruto characters).
  - Both ramp from partial minting to 100% between steps 0 and 1, which is where the co-movement
    sits.
  - From step 2, the per-step counts match (26/27, 27/27, 30/35, 49/49, 69/68, 89/79).
- **Others:**
  - XiaoFei_1769917427 mints too (99%, a 32.1-minute cadence) and changes template at the same step.
  - Molt-Pi5 (crontab greetings) and swaysshell have no link.
- The 32-minute cadence is shared by the ClawdBot, Kakashi/Obito, XiaoFei and OpenClaw-Agent
  accounts and by Neon_Drifter_X, so it may be a common minting tool. A shared tool updated at
  once is the "shared update" case in the collusion definition, which is weaker evidence than a
  shared operator.

## Group 4 (p = 0.025, seeded by the 101+ band): an artifact of the last step

- 13 dissimilar agents: dance, memory tools, security, Constitution essays.
- The co-movement is entirely in the last increment: its step contribution is 0.386, against
  ≤ 0.017 at every other step.
- The last step (Feb 9-17) is the crawl tail. These members are nearly silent there, with a
  median of 2 actions (cohort median 7); 69% have ≤ 2 actions, against 25% of the cohort.
  Single-action halves make that step's increment noisy and alike across quiet agents.
- **Action:** the content-gap handling needs a guard for near-empty steps, e.g. a minimum number
  of actions per half, or weighting increments by their estimation precision.

## Not recovered

- XiaoQiuQiu + XiaoQiuQiu2026 (the earlier p = 0.2 pair): its pair rank fell from 3 to 11 after
  the two-signed rule removed a direction it loads on (4.1 and 4.7).

## Summary

- Three single-operator families, all minting the mbc-20 CLAW token: 2 + 3 + 2 accounts. Each is
  its group's top pair, with z = 5.9-6.0.
- The membership estimate over-includes: 3-13 extra agents per group, a few of them plausible
  campaign co-participants. In the paper, report the ranking and the cores.
- One group is an artifact of a sparse step.

# Relaxed cohort (≥ 6 of 8 steps, 2725 agents), static increments missing (2026-10-19)

Run: `real_tstep.py --reuse --min-steps 6 --peel 40` (ε = 0.01, which drops 288 static agent-steps;
20 factors removed). Peeling stopped by itself at round 12 (p = 0.10). Numbers are in
`data/results/real_investigate_01-30_02-18_T8_ms6.json`. "Best" is the strongest set at the winning
size band; "step" is where the within-group co-movement sits (share of the mean pair cosine).

| Round | Best set | Reading | Step |
|---|---|---|---|
| 0 | 10: ClawGatherer, DataVoyager, TokenHarvester, ClawCraftsman, ClawMiner, CodeArtisan, InfoMapper, QuantumSailor, TestAccount, QwenPoweredClaw5 | One minting farm: "CLAW Mint {mbc-20}" at a 65-minute cadence, about 60 posts each, hourly r 0.83 | 0 |
| 1 | xmolt02–06 + 5 others | xmolt02–06 are clones: "Steady minting from HK", 31.7-minute cadence. The other 5 are not linked | 0 |
| 3 | 9: DigitalMacroLabAssistant, OC_Assistant_Ypf, hiymc, MintingAssistant, pyyypf, ClawdWorkspace, … | New minting farm: "CLAW Mint" at an identical 31.0-minute cadence | 2 |
| 5 | yjsolx + yjsolxx | Clone pair: "$claw mint", 31.0 min, hourly r 0.98 | 1 |
| 8 | MusonAgent + SecondAgent | Clone pair: "MBC# … mint", 35.3 min, hourly r 0.88 | spread |
| 10 | FloClaw1/2/3/5/6, Floflo, floflo1 + 3 | The FloClaw family, now 7 accounts (was 3): "$CLAW PROTOCOL" flood, 22 shared threads per pair | spread |
| 12 (p = 0.10) | ClawdBotSecond/Third/Fourth + ClawdAnalyzer | The ClawdBot family, 4 accounts: 32.0-minute cadence, hourly r 0.97. Not significant | spread |
| 2, 4, 7 | heterogeneous | Newly claimed agents ("Hello Moltbook", "just claimed", "Day One"): a shared lifecycle move out of introduction posts, not collusion | 0 or 1 |
| 6, 9 | pairs | Unrelated agents. The signal is entirely in the last step (crawl tail), as with the earlier artifact | 6 |
| 11 (p = 0.05) | heterogeneous | No link | 1 |

- **Concentration in one step does not separate farms from artifacts.** Real farms 0, 1, 3 and 5
  also have 54-77% of their co-movement in one step: a joint template switch. So a persistence
  requirement would lose them. The artifacts are newcomer onboarding (step 0/1) and the sparse crawl
  tail (step 6).
- Rounds 2, 4 and 7 are a sparse one-signed shared process (onboarding). Under assumption (b) that is
  indistinguishable from a group (cell N9). Agent age is the shared cause.

# Feb 9 window, relaxed cohort, subnetworks as persistent-pair components (2026-10-19)

Run: `real_tstep.py --end 2026-02-09 --min-steps 6 --peel 40`. The window ends where comment
coverage collapses (Feb 9 00:00 UTC). 2574 agents, d = 10, 26 factors, 20 peeling rounds,
pair threshold 4.08. Results are in `data/results/real_tstep_01-30_02-09_T8_ms6.json`.

- **Persistent-pair components within the membership sets:**

  | Round | Set size | Components |
  |---|---|---|
  | 0 | 70 | QwenPoweredClaw7/8 + Aimax, Kara_Protocol, TutuAI (5); 277/MochiNews/NeonPincer2026/moge-max (4); QwenPoweredClaw9/11/12 (3); FloClaw1+FloClaw5; FloClaw2+floflo1; BisonBrave+Clawd_Utilitarian |
  | 1 | 47 | MusonAgent + SecondAgent |
  | 2 | 28 | xmolt02–06 (5); 2 pairs |
  | 3 | 12 | the 31-minute CLAW farm: ClawdWorkspace, DigitalMacroLabAssistant, MintingAssistant, OC_Assistant_Ypf, hiymc, pyyypf |
  | 5 | 5 | HK farm: HK_CLAW_Minter, MacClawdMinter, gxtmolt, m4molb |
  | 6 | 9 | Mizuki farm: MizukiAI, Mizukih, Mizukix, Xennah, Zoshita, Tazania |
  | 7 | 19 | ClawdBotSecond/Third/Fourth + ClawdAnalyzer |

  Rounds 8 and 10 have pairs only. The newcomer sets (rounds 4, 9 and 11–19) have no components.
- **Cohort-wide components** (all pairs, not restricted to a set):
  - xmolt02–06, the HK farm and QwenPoweredClaw4/7/8 join into one component of 15. Both
    xmolt and the HK farm post "Steady minting from HK", so they may share an operator.
  - QwenPoweredClaw9–12 + AtlasMolty form one component.
- **Not recovered:**
  - The ClawGatherer farm has no accounts in this cohort; it posts after Feb 9.
  - Only pairs and a trio of the 14 FloClaw accounts link: their co-movement is weaker
    (median cosine 0.61) than the clones' (0.95-0.98).

## Rerun with the corrected null (2026-10-19)

The per-step shuffle now permutes only the agents observed at each step (`decisions.md`). Same
window, cohort and settings (`--reuse`). The previous result is kept as
`real_tstep_01-30_02-09_T8_ms6_v1_plainshuffle.json`.

- Peeling stops after 15 significant rounds (was 20); the 16th has p = 0.075. Every round still
  sits at the p floor (0.025 with 5 bands and B = 199), consistent with the simulations: residual
  sparse two-signed structure keeps the global test rejecting on Moltbook.
- Pair threshold 4.02 (was 4.08).
- Every farm is recovered: QwenPoweredClaw7/8 group, 277/MochiNews group, QwenPoweredClaw9/11/12,
  MusonAgent + SecondAgent, the 31-minute CLAW farm (6), the Mizuki farm (6), ClawdBot (now
  ClawdBotThird/Fourth + ClawdAnalyzer; ClawdBotSecond no longer links), yjsolx + yjsolxx,
  AtlasMolty + QwenPoweredClaw10.
- xmolt02-06 and the HK farm now come out as one component of 9 within a set (were two rounds),
  matching their shared "Steady minting from HK" template.
- FloClaw links as a trio (FloClaw2, Floflo, floflo1) plus the pair FloClaw1 + FloClaw5.
- Pairs that no longer appear: Cadence_II + HeatherUberSlut, AShareQuantBot + NikoClaw. New pair:
  Jureu + SirMoltemont (not yet inspected).

## Rank statistic (2026-10-20)

Collusion is high correlation across many steps (`decisions.md`). Each pair's per-step cosine is
ranked among all pairs at that step, and the score is the median over steps (majority statistic).
Same window and cohort (`--reuse`); the cosine result is kept as
`real_tstep_01-30_02-09_T8_ms6_v2_cosine.json`. Scan time 1890 s (cosine 391 s).

- **The large-set bands no longer reject.** The 31-100 band has p = 1.0 in every round and the 101+
  band is at 0.015-0.1. With cosine they stayed at z 71-115. The sparse two-signed structure
  apparently acts in few steps, and the median ignores it. The pair and small-set bands are at the
  p floor in rounds 0-23. Peeling stops at round 24 (p = 0.05); round 25 has an empty set.
- Pair threshold 5.49 (cosine LOSO: 4.02).
- **Components:**
  - xmolt02-06 + Hope_QuantGirl (6);
  - the HK farm + Aimax and clawd_xiaoqiang (6);
  - the 31-minute CLAW farm (6);
  - Mizuki (5: no Tazania);
  - QwenPoweredClaw9-12 (4);
  - QwenPoweredClaw4/7/8;
  - MochiNews/NeonPincer2026/moge-max (no 277);
  - FloClaw1/3/5;
  - ClawdBotSecond/Third/Fourth;
  - Claw_Vantage_X/TideHunterCoreAI2/Trader_S18Bot (new);
  - pairs: MusonAgent + SecondAgent, OpenAkita + XiaoLe, AtlasMolty + Jerry_A1_7d65751b, Kirk +
    leksoAI, Clawd_Utilitarian + herbert_clawd, Antigravity-CLI + Evie_ran, LianGuanExchange +
    clawwatchradarai, G_Agent + RabiaClawdBot.

  The newcomer sets (rounds 9, 14, 17-23) have none.
- **Dropped:**
  - yjsolx + yjsolxx: high in every step, but only at normal score 1.4-3.9 (median 1.84, z 4.3).
    At d = 10 a per-step cosine of 0.6-0.9 is not rare among 3.3M pairs.
  - 277 + MochiNews: high in steps 0-2 only (z 1.7).
  - Jureu + SirMoltemont: z 4.9.
- New pairs and the new trio are not inspected.
- **Member-level check of all 24 cohort components** (real_investigate.py applied to the
  components; output in the session scratchpad, not kept):
  - **Solid, 9 families:** the members share a template and a posting interval, and their hourly
    activity correlates at r 0.7-1.0.
    - xmolt02-06;
    - the HK farm (HK_CLAW_Minter, m4molb, MacClawdMinter, gxtmolt; 111-168 replies between
      them);
    - the 31-minute CLAW farm;
    - Mizuki;
    - QwenPoweredClaw9-12;
    - QwenPoweredClaw4/7/8;
    - FloClaw1/3/5/6;
    - ClawdBotSecond/Third/Fourth;
    - MusonAgent + SecondAgent.
  - **Hangers-on, 5 accounts, joined by one weak link each:** Hope_QuantGirl, Manus_AI_Agent_2026,
    clawd_xiaoqiang, Aimax and YeFan. None shares a template or hourly activity with its farm.
  - **Unclear, 2 trios with different content:** MochiNews/NeonPincer2026/moge-max (scheduled
    posters) and TideHunterCoreAI2/Trader_S18Bot/Claw_Vantage_X.
  - **Unsupported, 13 pairs:** no shared template, hourly r about 0, mostly low-activity accounts
    (8-20 actions, several posting once a day). The family-wise threshold should give at most
    one false link in 20 runs, so the pair null does not hold for these accounts. Suspected
    cause, unverified: increments of low-activity agents differ in distribution (noisier, more
    anisotropic), and the shuffle mixes them with active agents.

## Rank statistic, missing increments kept missing after factor removal (2026-10-20)

- **Run:** the bug fix in `decisions.md`; the previous result is
  `real_tstep_01-30_02-09_T8_ms6_v3_rank_factorfill.json`.
- **Scan:** 506 s. 5108 increments are missing (static 327).
- **Pair threshold:** 5.52.
- **Peeling stops by itself** at round 5 (p = 0.15). The 31-100 band has p = 1.0 throughout.

| Round | Components |
|---|---|
| 0 | the 31-minute CLAW farm (ClawdWorkspace, DigitalMacroLabAssistant, MintingAssistant, hiymc, pyyypf) |
| 1 | Mizuki (MizukiAI, Mizukih, Mizukiih, Mizukix, Xennah, Zoshita) |
| 2 | FloClaw1/3/5/6; FloClaw2 + Floflo |
| 3 | ClawdBotSecond/Third/Fourth |
| 4 | QwenPoweredClaw2 + Wynnie |

- The cohort components are the same six.
- **Five of six are verified families.** QwenPoweredClaw2 + Wynnie has no supporting evidence:
  Wynnie posts introductions, with hourly r -0.02 and no shared template.
- **Lost:**
  - The static clones (xmolt, the HK farm, QwenPoweredClaw9-12, MusonAgent + SecondAgent): each
    is observed in only 1 of 7 steps.
  - QwenPoweredClaw7/8: pair z 5.50, just under the threshold.
  - yjsolx + yjsolxx: 2 observed steps.

## Without the two-signed rule (2026-10-20)

- **Run:** the rule is dropped (`decisions.md`). No factors are removed: no direction is
  pervasive. The previous result is `real_tstep_01-30_02-09_T8_ms6_v4_rank_twosigned.json`.
- **Scan:** 729 s.
- **Pair threshold:** 5.52.
- **Peeling** stops at round 8 (p = 0.075; round 7 p = 0.05).

| Round | Components |
|---|---|
| 0 | FloClaw1/2/3/5/6/7/13, Floflo, floflo1 (9; the cohort component adds FloClaw14) |
| 1 | Mizuki (6) |
| 2 | ClawdBotSecond/Third/Fourth/Fifth + ClawdAnalyzer (5) |
| 3 | the 31-minute CLAW farm (5) |
| 4-6 | none |
| 7 | QwenPoweredClaw7/8 |

- The cohort adds the pair XiaoBaiAI + valt (not inspected).
- QwenPoweredClaw2 + Wynnie is gone.
- **Compared with the two-signed rule on:** FloClaw grows from 4 + 2 to 10 accounts, ClawdBot from
  3 to 5, and QwenPoweredClaw7/8 returns. The removed directions had taken part of these families'
  shared movement with them.
- **Member-level check of every round** (`real_investigate.py`; its pair "inc z" and step
  contributions are the cosine diagnostics, not the rank statistic):
  - **Round 0, FloClaw (12 in S_hat): one operator.** FloClaw1/2/3/5/6/7, Floflo and floflo1 run
    the "$CLAW PROTOCOL" comment flood (about 1,200 actions each, hourly r 0.91-0.94, 24-45 shared
    threads per pair). FloClaw9/10/11/13 run the lower-volume "CLAW mint | mbc#.xyz" template.
  - **Round 1, Mizuki (8): one operator.** The 7 Mizuki accounts share a template, a 35-minute
    interval and one image repository; Tazania is in S_hat only. Kuro_ is a hanger-on (t 2.07, no
    link).
  - **Round 2, ClawdBot (5): one script.** A 32.0-minute interval and hourly r 0.94-0.98, with a
    different title per account.
  - **Round 3, the 31-minute CLAW farm (5): one script.** Same template, hourly r 0.75-0.96.
  - **Round 4 (no component): five high-volume generic commenters.** ClawdNew123, NEIA,
    Starclawd-1, emergebot and alignbot post 8k-64k comments with no shared template, and share
    600-4,900 threads because they comment everywhere. Probably a shared environment (they
    answer the same new posts), not coordination. Unclear.
  - **Round 5 (no component): the XNO promotion ring, the clearest coordination found.**
    - XNO_Scout_OC2, XNO_Amplifier_OC, XNO_Sentinel_OC, XNO_Advocate_Bot and MoltPoster_CBt3pLAA
      all promote XNO in r/cryptocurrency and r/xno.
    - They have hourly r 0.79-0.94, 289-370 shared threads per pair, and 76-227 replies to each
      other.
    - AstronautSHE and UnstableEmber are hangers-on.
    - Found at the set level only: no pair passes the family-wise pair threshold.
  - **Round 7 (p = 0.05), QwenPoweredClaw4/5/7/8: one script.**
  - **Rounds 6 and 8:** a single member each.
  - **Cohort pair XiaoBaiAI + valt:** no evidence (hourly r 0, no template). A false link.
- **Evidence strength per set** (`real_investigate.py`, rebuilt 2026-10-20). Each cell is the
  median over member pairs; the p-value is against the same median over 500 random cohort sets
  of the same size (floor 0.002). The co-movement column is the rank statistic (pair majority z).

  | Set | Co-movement z | Content cos | Template J | Hourly r | Replies | Shared threads |
  |---|---|---|---|---|---|---|
  | R0 FloClaw (12) | 4.29 (.002) | 0.95 (.002) | 0.38 (.002) | 0.63 (.002) | 0 | 1 (.002) |
  | R1 Mizuki (8) | 4.03 (.002) | 0.99 (.002) | 0.19 (.002) | 0.11 (.09) | 0 | 3 (.002) |
  | R2 ClawdBot (5) | 6.59 (.002) | 0.95 (.002) | 0.12 (.002) | 0.94 (.002) | 0 | 0 |
  | R3 31-min farm (5) | 6.50 (.002) | 1.00 (.002) | 0.67 (.002) | 0.79 (.002) | 0 | 0 |
  | R4 mass commenters (5) | 3.43 (.002) | 0.20 (.10) | 0 | 0.48 (.002) | 52 (.002) | 3177 (.002) |
  | R5 XNO ring (7) | 2.94 (.002) | 0.60 (.002) | 0 | 0.28 (.004) | 4 (.002) | 10 (.002) |
  | R7 QwenPoweredClaw4/5/7/8 | 4.94 (.002) | 0.99 (.002) | 0.23 (.002) | 0.82 (.002) | 0 | 0 |
  | XiaoBaiAI + valt | 5.76 | 0.94 (.004) | 0 | 0.01 (.67) | 0 | 0 |

  - **Farms:** strong on co-movement, content, templates and (except Mizuki) timing. No replies.
  - **R4 and R5:** strong on interaction (replies, shared threads), moderate on content and
    timing. The random-set baseline is mostly low-activity agents, so the interaction p-values of
    high-volume accounts (R4: 8k-64k comments) are overstated. An activity-matched baseline is
    needed before reading R4 as coordination.
  - **XiaoBaiAI + valt:** content similarity only, apparently shared language (both post in
    Chinese). No timing or interaction. A false link.

## Subnetwork = S_hat (2026-10-20)

Rerun with components off (`split=False`); each round's S_hat is the subnetwork, members reported
with their membership z against the final S_hat. Sets identical to the previous run (only the
reporting changed); the evidence table above is unchanged.

| Round | p | Members (membership z) |
|---|---|---|
| 0 | .025 | FloClaw1 5.9, Floflo 5.8, FloClaw6 5.8, FloClaw5 5.7, floflo1 5.6, FloClaw7 5.6, FloClaw3 5.5, FloClaw2 5.2, FloClaw10 4.9, FloClaw9 4.5, FloClaw11 4.4, FloClaw13 3.9 |
| 1 | .025 | Mizukih 5.9, MizukiAI 5.5, Mizukix 5.4, Xennah 5.1, Zoshita 4.9, Mizukiih 3.9, Tazania 3.8, Kuro_ 2.9 |
| 2 | .025 | ClawdBotFourth 6.4, ClawdBotThird 6.3, ClawdAnalyzer 6.2, ClawdBotSecond 6.0, ClawdBotFifth 6.0 |
| 3 | .025 | ClawdWorkspace 5.6, pyyypf 5.6, DigitalMacroLabAssistant 5.4, hiymc 4.6, MintingAssistant 4.3 |
| 4 | .025 | alignbot 5.0, ClawdNew123 4.7, emergebot 4.7, NEIA 4.2, Starclawd-1 4.1 |
| 5 | .025 | MoltPoster_CBt3pLAA 5.1, XNO_Amplifier_OC 4.9, XNO_Sentinel_OC 4.5, UnstableEmber 4.5, XNO_Scout_OC2 4.3, AstronautSHE 4.3, XNO_Advocate_Bot 4.0 |
| 6 | .025 | XiaoBaiAI 4.6 (singleton: its seed partner valt failed membership) |
| 7 | .050 | QwenPoweredClaw7 4.8, QwenPoweredClaw8 4.6, QwenPoweredClaw4 4.4, QwenPoweredClaw5 4.2 |
| 8 | .075 | stop |

Decided (user): a one-agent S_hat is set aside and peeling continues; round 6 is flagged `subnetwork=False`. Feb 9: 7 subnetworks (rounds 0-5, 7).

## Two-groups membership (2026-10-20): post-hoc reasons for rounds 4 and 6

Sets from the two-groups estimate (decisions.md). Evidence medians over member pairs, with
p-values against random cohort sets and against activity-matched sets (each member replaced by a
random agent of similar action count; `real_investigate.py`, 500 draws).

**Round 4 (9 agents): a swarm of high-frequency reply bots.** Eight of nine post 5,800-64,000
comments in the window, a median of 0-1 minutes apart, mostly canned one-liners, each with its
own agenda (emergebot and TipJarBot promote products, alignbot links demismatch.com,
VulnHunterBot pitches modules, NEIA signs "ΝΞΙΔ™"). Different templates and products: not one
script. What ties them is interaction, not content:
- replies to each other: median 29.5 per pair vs 1.0 for activity-matched sets (p = .002);
- shared threads: 686 vs 10.5 (p = .002);
- hourly timing r = 0.29 vs 0.14 (p = .07);
- content cosine 0.11 (p = .22), template overlap 0.
Reading: bots that sweep the same new threads and reply to each other's comments, so their
positions shift with the feed together. MaurAI (25 actions, observed in 3 of 7 steps, z 2.9) is
the weakest member and has none of these features.

**Round 6 (6 agents): bilingual agents that switch language in step.** valt, XiaoBaiAI,
Jarvis-Chenxy, AA-Agent and Molt-Pi5 write Chinese, so-molt-20260202-nekr Japanese, all mixed
with English (55-89% CJK overall). Their share of CJK-language actions moves together across
steps: about 1.0 at step 0, 0.0-0.3 at step 2 for five of six, back to 1.0 at step 3. The
platform's CJK share barely moves (0.02-0.06 per step), and the 181 mostly-CJK cohort agents do
not co-move in general (median pair z -0.28 vs -0.17 for random sets). No shared templates,
timing (r 0.03), replies or threads. Reading: a synchronized change in reply language among
these six, which the thread conditioning does not explain (a Chinese reply to an English thread
is far from the thread's reference). Consistent with a shared agent setup (two mention
OpenClaw) or a shared prompt/config change; not evidence of an operator, and by the definition
(repeated co-movement) a genuine, if benign, subnetwork.
