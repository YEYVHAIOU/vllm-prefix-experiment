# V/U/C/F Benchmark Run Order

This pack adds only benchmark configs and shell wrappers. It does not modify
`source/`, `deployment/`, or the existing Python replay scripts.

## Case definitions

- V: FCFS, UCM OFF.
- U: FCFS, UCM ON, eager WHEN, full WHAT, TieredStore WHERE.
- C: Continuum scheduler, UCM OFF.
- F: Continuum scheduler, UCM ON, joint/cost_full WHEN,
  frontier_tail K=4 WHAT, TieredStore WHERE.

## Required run order

1. `benchmark/scripts/smoke_four_cases.sh`
2. `benchmark/scripts/run_c76_pilot.sh`
3. `benchmark/scripts/run_vucf_capacity.sh`
4. If every case remains stable at C128, build higher-concurrency trajectory
   workloads and rerun capacity with e.g.
   `CAPACITY_LEVELS="160 192" benchmark/scripts/run_vucf_capacity.sh`.
5. Only after the maximum-stable-concurrency study is closed, run the final
   full-EnvBench campaign.

## Capacity semantics

The capacity campaign preserves the historical capacity settings:

- duration_scale = 0.01
- max_sleep_seconds = 2
- httpclient_per_lane
- balanced trajectory lanes
- request timeout = 300 s

A stable case requires all expected requests to succeed and no server-side
OOM/EngineCore/runtime failure.

The current C96 and C128 workloads must be copied into `benchmark/workloads/`
before the capacity campaign is run.
