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
