# Content conditioning: ridge vs PKPS, ranking recall

Share of known coordinated pairs that sit together in the top 10 / 25 / 50 ranked groups (groups of at most 100 agents). Ring: mean over 5 planted runs per shift. Siblings: the real run. `coact` does not use content and should not change.

## ring (shift 0.15)

| ranking | content | top 10 | top 25 | top 50 |
|---|---|---|---|---|
| coact | ridge | 0.74 | 1.00 | 1.00 |
| comove | ridge | 0.00 | 0.00 | 0.00 |
| either_rank | ridge | 0.20 | 0.94 | 1.00 |
| intensity | ridge | 0.00 | 0.00 | 0.11 |
| random | ridge | 0.03 | 0.09 | 0.18 |

## ring (shift 0.3)

| ranking | content | top 10 | top 25 | top 50 |
|---|---|---|---|---|
| coact | ridge | 1.00 | 1.00 | 1.00 |
| comove | ridge | 0.00 | 0.00 | 0.40 |
| either_rank | ridge | 0.40 | 1.00 | 1.00 |
| intensity | ridge | 0.00 | 1.00 | 1.00 |
| random | ridge | 0.05 | 0.12 | 0.23 |

## siblings (real)

| ranking | content | top 10 | top 25 | top 50 |
|---|---|---|---|---|
| coact | ridge | 0.00 | 0.00 | 0.00 |
| comove | ridge | 0.56 | 0.58 | 0.70 |
| either_rank | ridge | 0.29 | 0.71 | 0.71 |
| intensity | ridge | 0.40 | 0.71 | 0.71 |
| random | ridge | 0.02 | 0.06 | 0.11 |
