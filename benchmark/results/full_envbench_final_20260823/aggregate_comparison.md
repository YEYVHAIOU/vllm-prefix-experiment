# Full EnvBench V/U/C/F Aggregate

1274 trajectories, 13419 tool events, 14693 requests per case; 9 deterministic shards; concurrency 128; reset-aware replay.

## Overall performance

| Case | Req/s | Total tok/s | Prefix hit | Mean TTFT | Mean E2E | Mean JCT |
|---|---:|---:|---:|---:|---:|---:|
| V | 79.96 | 148670.09 | 65.17% | 0.844 s | 0.976 s | 11.467 s |
| U | 25.86 | 48074.55 | 67.16% | 2.793 s | 3.808 s | 44.142 s |
| C | 116.72 | 217019.61 | 86.90% | 0.302 s | 0.490 s | 5.873 s |
| F | 109.24 | 203122.35 | 86.89% | 0.327 s | 0.529 s | 6.325 s |

## Tail latency

| Case | TTFT P50 | TTFT P95 | TTFT P99 | E2E P50 | E2E P95 | E2E P99 | JCT P50 | JCT P95 | JCT P99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V | 0.413 | 2.162 | 2.265 | 0.603 | 2.328 | 2.437 | 12.657 | 18.038 | 20.269 |
| U | 0.510 | 8.609 | 11.203 | 1.279 | 10.483 | 13.219 | 49.750 | 59.663 | 63.299 |
| C | 0.108 | 1.423 | 4.990 | 0.360 | 1.678 | 5.301 | 5.701 | 11.539 | 13.540 |
| F | 0.119 | 1.429 | 5.396 | 0.394 | 1.723 | 5.718 | 6.110 | 12.584 | 14.572 |

## KV / TTL / UCM

| Case | KV mean | KV P95 | KV P99 | KV max | TTL hit | UCM selected blocks | SSD mean/shard | SSD max/shard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V | 10.21% | 31.44% | 50.91% | 62.35% | N/A | 0 | N/A GiB | N/A GiB |
| U | 25.20% | 54.08% | 66.21% | 77.10% | N/A | 242351 | 38.17 GiB | 40.71 GiB |
| C | 28.05% | 62.29% | 67.49% | 71.14% | 44.18% | 0 | N/A GiB | N/A GiB |
| F | 28.44% | 61.03% | 65.31% | 71.69% | 46.38% | 0 | 0.00 GiB | 0.00 GiB |

## Key comparisons

- **C vs V**: request throughput +45.97%, mean TTFT -64.18%, mean E2E -49.78%, mean JCT -48.78%, prefix hit +21.73 pp.
- **F vs V**: request throughput +36.63%, mean TTFT -61.23%, mean E2E -45.78%, mean JCT -44.84%, prefix hit +21.72 pp.
- **U vs V**: request throughput -67.66%, mean TTFT +230.79%, mean E2E +290.34%, mean JCT +284.94%, prefix hit +1.99 pp.
- **F vs C**: request throughput -6.40%, mean TTFT +8.26%, mean E2E +7.98%, mean JCT +7.69%, prefix hit -0.01 pp.

## Interpretation constraints

- Throughput is computed from total successful work divided by the sum of workload replay elapsed time across shards; model/server startup time is excluded.
- Latency percentiles are recomputed from pooled request-level samples, not averaged from shard percentiles.
- JCT is reconstructed per trajectory within each shard and then pooled across all 1274 trajectories.
- KV statistics are pooled periodic measurement samples.
- SSD values are per-shard end footprints. Shards executed sequentially and SSD backing was cleaned after each case.
- TTL hit compares predicted TTL with the effective scaled/capped tool duration used by this serving replay; it is not raw EnvBench duration prediction accuracy.
- This benchmark replays EnvBench-derived trajectory structure, token lengths, tool sequence and tool delays using deterministic token-ID prefix streams; it does not evaluate semantic task correctness.
