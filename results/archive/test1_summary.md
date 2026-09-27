# Test 1 Summary — Core Scheduler vs Baselines
CI method: bootstrap 2000 resamples, 95%

## Scenario: background

| Algorithm | Intercept Rate (mean±CI) | Worst | Avg Reward (mean±CI) | Worst | Wilcoxon p vs WIQL |
|---|---|---|---|---|---|
| round_robin | 0.304 [0.285, 0.322] | 0.146 | 0.142 [0.126, 0.158] | 0.054 | — |
| random | 0.306 [0.287, 0.324] | 0.147 | 0.142 [0.127, 0.158] | 0.057 | — |
| wiql_ucb | 0.287 [0.210, 0.364] | 0.009 | 0.126 [0.095, 0.159] | 0.004 | — |
| wiql_ucb_no_bias | 0.503 [0.455, 0.550] | 0.193 | 0.233 [0.206, 0.262] | 0.079 | N/A |

## Scenario: periodic

| Algorithm | Intercept Rate (mean±CI) | Worst | Avg Reward (mean±CI) | Worst | Wilcoxon p vs WIQL |
|---|---|---|---|---|---|
| round_robin | 0.319 [0.303, 0.333] | 0.217 | 0.123 [0.111, 0.135] | 0.047 | — |
| random | 0.320 [0.304, 0.336] | 0.214 | 0.123 [0.111, 0.135] | 0.046 | — |
| wiql_ucb | 0.310 [0.226, 0.399] | 0.023 | 0.113 [0.084, 0.144] | 0.007 | — |
| wiql_ucb_no_bias | 0.511 [0.461, 0.561] | 0.144 | 0.197 [0.171, 0.224] | 0.078 | N/A |

## Scenario: freq_agile

| Algorithm | Intercept Rate (mean±CI) | Worst | Avg Reward (mean±CI) | Worst | Wilcoxon p vs WIQL |
|---|---|---|---|---|---|
| round_robin | 0.316 [0.303, 0.328] | 0.201 | 0.231 [0.219, 0.243] | 0.131 | — |
| random | 0.317 [0.304, 0.329] | 0.202 | 0.232 [0.219, 0.243] | 0.132 | — |
| wiql_ucb | 0.303 [0.248, 0.359] | 0.064 | 0.219 [0.181, 0.258] | 0.050 | — |
| wiql_ucb_no_bias | 0.456 [0.426, 0.490] | 0.286 | 0.333 [0.310, 0.357] | 0.218 | N/A |

