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
