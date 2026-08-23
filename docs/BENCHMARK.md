# Benchmark Methodology and Final Results

## 1. Evaluation Overview

This document records the final benchmark methodology and results for the integrated vLLM + Continuum + UCM system.

Four cases are evaluated under the same serving configuration:

| Case | System |
|---|---|
| V | vanilla vLLM |
| U | vLLM + UCM |
| C | vLLM + Continuum |
| F | vLLM + Continuum + UCM |

The formal final evaluation is the **Full Runtime-Eligible EnvBench** serving-system replay.

## 2. Hardware and Serving Configuration

Formal experiments were executed on AutoDL with:

- GPU: NVIDIA GeForce RTX 4090, 24564 MiB visible memory
- CPU: 16 cores
- Host memory: approximately 120 GB
- Python: 3.12.3
- PyTorch: 2.8.0+cu129
- CUDA toolkit: 12.8
- GCC: 11.2
- vLLM build: `0.1.dev10+g05f00f8a8`
- Model: Qwen/Qwen3-0.6B

Frozen serving parameters:

- server `max_model_len = 4096`
- benchmark maximum input length = 4000 tokens
- maximum generation = 8 tokens
- `gpu_memory_utilization = 0.80`
- `swap_space = 1 GiB`
- concurrency = 128
- HTTP mode = `httpclient_per_lane`
- balanced lanes
- duration scale = 0.01
- maximum replay sleep = 2 s
- reset-aware prefix replay = enabled

The 4096-token value is the server-side sequence-length limit used for this experiment; it is not a claim about the intrinsic context limit of Qwen3-0.6B.

## 3. System Variants

### V — vanilla vLLM

- FCFS scheduler
- Continuum disabled
- UCM disabled
- GPU-only KV cache

### U — vLLM + UCM

- FCFS scheduler
- UCM enabled
- WHEN: eager
- WHAT: full
- WHERE: TieredStore
- DRAM tier: 128 MiB
- SSD secondary backing

### C — vLLM + Continuum

- Continuum scheduler
- Dynamic TTL enabled
- UCM disabled

### F — Full System

- Continuum scheduler
- Dynamic TTL enabled
- UCM enabled
- WHEN: joint / `cost_full`
- WHAT: Frontier-Tail, K=4
- WHERE: TieredStore
- DRAM tier: 128 MiB
- SSD secondary backing

The final architecture is:

**Dynamic TTL + cost-aware WHEN + Frontier-Tail WHAT + TieredStore WHERE**

Frontier-Tail should be described as **Sparse External KV Backing / Frontier-Tail Partial Retention**, not as official UCM Sparse Attention.

## 4. Workloads

### 4.1 High-Contention Pressure Workload

`benchmark/workloads/envbench_high_contention_balanced_min3500.csv`

This workload is pressure-enhanced. Its serving token length is defined as:

`pressure_tokens = max(original_input_tokens, 3500)`

Therefore it must not be described as a dataset whose original prompts are all at least 3500 tokens long.

It was used for C76/C96/C128 stress and capacity experiments. The correct statement is **validated stable concurrency of at least 128**, not “maximum concurrency is 128”.

### 4.2 Full Runtime-Eligible EnvBench

Formal workload:

`benchmark/workloads/envbench_full_runtime_eligible_max4000.csv`

Audit:

`benchmark/workloads/envbench_full_runtime_eligible_max4000.audit.json`

Construction rules:

1. start from the normalized eligible single-call EnvBench trace;
2. preserve complete trajectories;
3. accept only trajectories whose every event has valid duration and `1 <= input_tokens <= 4000`;
4. reject the whole trajectory if any event exceeds 4000 tokens;
5. do not truncate prompts or turns;
6. do not apply min3500 pressure padding;
7. do not use high-contention scoring or random sampling.

Final workload:

| Metric | Value |
|---|---:|
| Source trajectories | 3677 |
| Source eligible events | 54078 |
| Accepted trajectories | 1274 |
| Accepted tool events | 13419 |
| Expected requests | 14693 |
| Accepted source-event input tokens | 24,822,191 |
| Rejected trajectories | 2403 |

Accepted input tokens: min 433, mean 1849.78, median 1755, P90 3142, P95 3429, P99 3804, max 3997.

## 5. Reset-Aware Replay

Among the 1274 accepted trajectories, 622 are monotonic in input-token length and 652 contain exactly one input-token decrease. Raw inspection showed these decreases correspond to real EnvBench search-to-build stage transitions.

When a decrease occurs, the final runner starts a new deterministic token stream and increments `context_segment`. This prevents the new build-stage prompt from being falsely treated as a prefix of the old search-stage prompt.

The formal campaign records:

- `reset_aware_prefix = true`
- `context_reset_policy = new deterministic token stream after input-token decrease`

## 6. Replay Semantics

The benchmark does not reconstruct original EnvBench prompt text. It replays EnvBench-derived trajectory structure using deterministic token-ID prefix streams matching the recorded token lengths.

It preserves trajectory structure, tool order, tool delays, input-token lengths, prefix-growth relationships, and search-to-build resets.

It does **not** evaluate semantic task correctness, code-generation quality, original-language understanding, or EnvBench end-task success.

The correct description is **EnvBench-derived serving-system replay**.

## 7. Formal Campaign Completeness

The Full workload was split deterministically into 9 whole-trajectory shards with capacities:

`142, 142, 142, 142, 142, 141, 141, 141, 141`

Every shard was evaluated with `V -> U -> C -> F` at concurrency 128.

Final validation:

```text
V requests=14693 success=14693 failed=0
U requests=14693 success=14693 failed=0
C requests=14693 success=14693 failed=0
F requests=14693 success=14693 failed=0
cases=36
total_requests_all_cases=58772
FULL_36_CASE_VALIDATION=PASS
```

## 8. Aggregation Rules

Final aggregate artifacts are under:

`benchmark/results/full_envbench_final_20260823/`

Aggregation rules:

- throughput = total successful work / sum of replay elapsed time across shards;
- TTFT/TPOT/E2E percentiles are recomputed from pooled request-level samples;
- JCT is reconstructed per `shard + job_id` and pooled across all 1274 trajectories;
- Prefix Cache hit rate = summed hit tokens / summed query tokens;
- TTL counts are summed, with TTL seconds derived from `tool_events.csv`;
- KV utilization is pooled across periodic KV sample rows;
- SSD footprints are reported per shard and are not summed as simultaneous capacity.

## 9. Final Full EnvBench Results

### 9.1 Overall Performance

| Case | Req/s | Total tok/s | Prefix Hit | Mean TTFT | Mean E2E | Mean JCT |
|---|---:|---:|---:|---:|---:|---:|
| V | 79.96 | 148,670.09 | 65.17% | 0.844 s | 0.976 s | 11.467 s |
| U | 25.86 | 48,074.55 | 67.16% | 2.793 s | 3.808 s | 44.142 s |
| C | **116.72** | **217,019.61** | **86.90%** | **0.302 s** | **0.490 s** | **5.873 s** |
| F | 109.24 | 203,122.35 | 86.89% | 0.327 s | 0.529 s | 6.325 s |

### 9.2 Tail Latency (seconds)

| Case | TTFT P50 | TTFT P95 | TTFT P99 | E2E P50 | E2E P95 | E2E P99 | JCT P50 | JCT P95 | JCT P99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V | 0.413 | 2.162 | 2.265 | 0.603 | 2.328 | 2.437 | 12.657 | 18.038 | 20.269 |
| U | 0.510 | 8.609 | 11.203 | 1.279 | 10.483 | 13.219 | 49.750 | 59.663 | 63.299 |
| C | 0.108 | 1.423 | 4.990 | 0.360 | 1.678 | 5.301 | 5.701 | 11.539 | 13.540 |
| F | 0.119 | 1.429 | 5.396 | 0.394 | 1.723 | 5.718 | 6.110 | 12.584 | 14.572 |

### 9.3 GPU KV Cache Utilization

| Case | Mean | P95 | P99 | Max |
|---|---:|---:|---:|---:|
| V | 10.21% | 31.44% | 50.91% | 62.35% |
| U | 25.20% | 54.08% | 66.21% | 77.10% |
| C | 28.05% | 62.29% | 67.49% | 71.14% |
| F | **28.44%** | 61.03% | 65.31% | **71.69%** |

Recommended Full-System reporting:

> Under Full EnvBench at concurrency 128, GPU KV Cache utilization is 28.44% on average, 61.03% at P95, 65.31% at P99, with a peak of 71.69%.

### 9.4 Dynamic TTL

| Case | TTL Hit Rate |
|---|---:|
| V | N/A |
| U | N/A |
| C | 44.18% |
| F | 46.38% |

These values compare against effective scaled/capped replay duration and are not raw EnvBench duration-prediction accuracy.

### 9.5 UCM / SSD

U:

- approximately 242,351 selected KV blocks across all 9 shards;
- mean SSD end footprint: approximately 38.17 GiB per shard;
- maximum SSD end footprint: approximately 40.71 GiB per shard.

F:

- `cost_full` selected blocks: 0;
- Tier CREATE/DUMP/LOAD: 0;
- SSD backing: metadata-level only, effectively 0 GiB.

Thus the Full System did not perform actual external KV migration on this workload.

## 10. Key Comparisons

### C vs V

- request throughput: **+45.97%**
- mean TTFT: **-64.18%**
- mean E2E: **-49.78%**
- mean JCT: **-48.78%**
- Prefix Cache hit rate: **+21.73 percentage points**

### U vs V

- request throughput: **-67.66%**
- mean TTFT: **+230.79%**
- mean E2E: **+290.34%**
- mean JCT: **+284.94%**
- Prefix Cache hit rate: only **+1.99 percentage points**

### F vs V

- request throughput: **+36.63%**
- mean TTFT: **-61.23%**
- mean E2E: **-45.78%**
- mean JCT: **-44.84%**
- Prefix Cache hit rate: **+21.72 percentage points**

### F vs C

- request throughput: **-6.40%**
- mean TTFT: **+8.26%**
- mean E2E: **+7.98%**
- mean JCT: **+7.69%**
- Prefix Cache hit rate: **-0.01 percentage points**

No statistical-significance claim is made for the F-vs-C difference.

## 11. Final Interpretation

1. **Continuum is the primary source of the measured performance gain.** It raises Prefix Cache reuse from about 65% to about 87% and substantially improves throughput and mean latency.
2. **Naive eager/full externalization is expensive.** U creates large SSD traffic and severe latency/throughput degradation while adding little Prefix Cache reuse.
3. **Cost-aware UCM prevents harmful migration.** In F, `cost_full` evaluates candidates but selects zero blocks, avoiding the eager/full behavior observed in U.
4. **Frontier-Tail positive external-I/O benefit is not demonstrated by this campaign.** Frontier-Tail is integrated and available, but the cost-aware WHEN policy correctly chose not to invoke external migration for the evaluated Full workload.

The defensible Full-System conclusion is:

> The integrated Continuum + UCM system retains most of the performance gain introduced by Continuum while using cost-aware policy logic to avoid external KV migration when estimated migration/reload cost exceeds expected benefit.

## 12. Reproducibility Artifacts

- `benchmark/workloads/envbench_full_runtime_eligible_max4000.csv`
- `benchmark/workloads/envbench_full_runtime_eligible_max4000.audit.json`
- `benchmark/workloads/full_shards_9/manifest.tsv`
- `benchmark/results/full_envbench_final_20260823/manifest.tsv`
- `benchmark/results/full_envbench_final_20260823/aggregate_summary.json`
- `benchmark/results/full_envbench_final_20260823/aggregate_summary.tsv`
- `benchmark/results/full_envbench_final_20260823/aggregate_comparison.md`
- `benchmark/results/full_envbench_final_20260823/pooled_requests/`
- `benchmark/results/full_envbench_final_20260823/pooled_jct/`

Primary scripts:

- `benchmark/scripts/build_full_envbench_workload.py`
- `benchmark/scripts/build_full_envbench_shards.py`
- `benchmark/scripts/run_envbench_runtime_concurrent.py`
- `benchmark/scripts/run_envbench_runtime_httpclient.py`
- `benchmark/scripts/run_case_workload.sh`
- `benchmark/scripts/run_full_envbench_vucf.sh`
- `benchmark/scripts/summarize_case.py`
- `benchmark/scripts/aggregate_full_envbench.py`

## 13. Reporting Constraints

Use these descriptions:

- “Full Runtime-Eligible EnvBench”
- “EnvBench-derived serving-system replay”
- “server max sequence length = 4096 tokens”
- “maximum benchmark input = 4000 tokens”
- “validated stable concurrency of at least 128”

Do not claim:

- Qwen3-0.6B intrinsically has only a 4096-token context window;
- min3500 contains only naturally >=3500-token prompts;
- concurrency 128 is the maximum possible concurrency;
- TTL hit rate is raw EnvBench duration-prediction accuracy;
- Frontier-Tail external I/O improved Full EnvBench performance;
- sequential shard SSD footprints should be summed as simultaneous disk usage;
- deterministic token-ID replay is semantic EnvBench task evaluation.
