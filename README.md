# vLLM + Continuum + UCM

**Cost-aware KV Cache management for long-context, multi-turn Agent serving**

This repository contains the deployment, benchmark, workload-construction, and final evaluation artifacts for a research integration of **vLLM**, **Continuum-style temporal KV retention**, and **Unified Cache Management (UCM)**.

The project targets multi-turn Agent workloads in which long reusable prefixes are separated by tool-call gaps. Under high concurrency, these gaps can cause otherwise reusable KV Cache blocks to be evicted before the next turn arrives.

> This repository is a research integration artifact and is **not** an official vLLM, Continuum, UCM, or EnvBench distribution.

---

## 1. Project overview

The final system combines two complementary mechanisms:

- **Continuum / Dynamic TTL** — predicts how long reusable KV blocks should remain protected from eviction.
- **UCM** — controls external KV migration through:
  - **WHEN**: whether migration is worthwhile,
  - **WHAT**: which blocks should be migrated,
  - **WHERE**: where migrated blocks should be stored.

The final configuration is:

```text
Dynamic TTL
    +
cost_full WHEN
    +
Frontier-Tail WHAT (K=4)
    +
TieredStore WHERE (DRAM + SSD)
```

Conceptually:

```text
Agent request
     |
     v
GPU KV Cache
     |
     +-- Continuum / Dynamic TTL
     |      `-- Which reusable blocks should remain protected?
     |
     `-- UCM
            +-- WHEN  : Is migration cost-effective?
            +-- WHAT  : Which KV blocks should migrate?
            `-- WHERE : DRAM / SSD
```

`Frontier-Tail` in this project is a project-specific sparse external KV backing / partial-retention policy. It should **not** be confused with UCM Sparse Attention.

---

## 2. Repository layout

The project is split into three Git repositories.

```text
workspace/
├── vllm-prefix-experiment/
├── vllm-continuum/
└── unified-cache-management-continuum/
```

### Main experiment repository

This repository:

```text
vllm-prefix-experiment/
├── README.md
├── docs/
├── deployment/
│   ├── configs/
│   └── scripts/
├── benchmark/
│   ├── configs/
│   ├── scripts/
│   ├── workloads/
│   └── results/
└── manifests/
```

It contains:

- deployment configuration and lifecycle scripts,
- V/U/C/F benchmark runners,
- EnvBench-derived serving workloads,
- final aggregate benchmark results,
- source and environment manifests,
- compatibility and provenance documentation.

Runtime logs, model weights, virtual environments, frozen Golden Runtime copies, and large raw request traces are intentionally excluded from Git.

### Source repositories

Modified vLLM + Continuum source:

- https://github.com/YEYVHAIOU/vllm-continuum

Modified UCM source:

- https://github.com/YEYVHAIOU/unified-cache-management-continuum

---

## 3. Frozen source identity

### vLLM + Continuum

```text
branch : joint-offload-v1
commit : 6e7d571b831e6e4b82f1b2f8228cc84e9a0261a2
tag    : vllm-continuum-final-20260820
```

### UCM

```text
branch : joint-offload-v1
commit : be59181f50e515496a7f19a38178c8c2a05cf251
tag    : ucm-continuum-final-20260822
```

The UCM commit above includes the final fix that initializes KV caches before the empty-load fast path.

See:

- `manifests/source_identity.txt`
- `manifests/final_release_identity.txt`
- `docs/SOURCE_PROVENANCE.md`

---

## 4. Final tested environment

The final formal benchmark was run on an AutoDL instance with:

```text
GPU                 : NVIDIA GeForce RTX 4090
GPU memory          : 24564 MiB
CPU                 : 16 vCPU
Host memory         : ~120 GB
Python              : 3.12
PyTorch             : 2.8.0+cu129
vLLM package        : 0.1.dev10+g05f00f8a8
CUDA_HOME tested    : /usr/local/cuda-12.8
Model               : Qwen/Qwen3-0.6B
Server max model len: 4096
Max benchmark input : 4000 tokens
Max output          : 8 tokens
GPU memory util     : 0.80
Swap space          : 1 GiB
```

`MAX_MODEL_LEN=4096` is the serving-experiment cap used in this study, not the intrinsic maximum context length of the model.

Exact environment snapshots are in `manifests/`.

---

## 5. Benchmark cases

The final comparison uses four serving configurations.

| Case | Scheduler | Continuum | UCM |
|---|---|---:|---|
| **V** | FCFS | Off | Off |
| **U** | FCFS | Off | eager / full |
| **C** | Continuum | On | Off |
| **F** | Continuum | On | cost-aware |

Interpretation:

- **V**: vLLM-style baseline.
- **U**: baseline scheduling plus eager/full external KV migration.
- **C**: Continuum / Dynamic TTL without UCM migration.
- **F**: final combined system using Continuum and cost-aware UCM.

---

## 6. Formal Full EnvBench-derived benchmark

The final formal evaluation uses a **Full Runtime-Eligible EnvBench-derived serving replay**.

```text
Accepted trajectories : 1274
Tool events           : 13419
Requests per case     : 14693
Cases                 : V / U / C / F
Deterministic shards  : 9
Concurrency           : 128
Case runs             : 36
Validation            : 36 / 36 PASS
Total requests        : 58772
```

The accepted workload contains inputs up to 4000 tokens.

### Important scope limitation

This is an **EnvBench-derived serving-system replay** using deterministic token-ID prefix streams.

It evaluates:

- serving latency,
- throughput,
- KV Cache reuse,
- KV occupancy,
- external KV migration behavior,
- scheduler / cache-policy effects.

It does **not** evaluate semantic EnvBench task accuracy.

Therefore, these results should not be interpreted as an EnvBench agent-task leaderboard score.

---

## 7. Final results

Aggregate results from the formal Full Runtime-Eligible EnvBench-derived benchmark:

| Case | Req/s | Prefix Hit | Mean TTFT | Mean E2E | Mean JCT |
|---|---:|---:|---:|---:|---:|
| **V** | 79.96 | 65.17% | 0.844 s | 0.976 s | 11.467 s |
| **U** | 25.86 | 67.16% | 2.793 s | 3.808 s | 44.142 s |
| **C** | **116.72** | **86.90%** | **0.302 s** | **0.490 s** | **5.873 s** |
| **F** | 109.24 | 86.89% | 0.327 s | 0.529 s | 6.325 s |

The complete aggregate files are under:

```text
benchmark/results/full_envbench_final_20260823/
```

including:

- `aggregate_summary.json`
- `aggregate_summary.tsv`
- `aggregate_comparison.md`
- `manifest.tsv`

---

## 8. Main findings

### Continuum is the primary performance gain

Compared with V:

```text
Throughput:
79.96 -> 116.72 req/s

Prefix hit rate:
65.17% -> 86.90%
```

This corresponds to approximately a **45.97% throughput increase** for C over V in the final formal benchmark.

Dynamic TTL improves the probability that reusable prefixes survive tool-call gaps and remain available for subsequent turns.

### Eager/full UCM migration is harmful in this workload

U performs substantially worse than V:

```text
79.96 -> 25.86 req/s
```

The main cause is large external KV migration overhead.

Across the final U runs, UCM selected a large number of blocks for migration and generated substantial SSD traffic.

This result shows that external KV capacity alone is not sufficient: migration must be selective enough for the transfer cost to be justified.

### Cost-aware UCM avoids harmful migration

In the formal Full EnvBench-derived F runs, the final cost-aware policy selected:

```text
UCM selected blocks : 0
Tier CREATE         : 0
Tier DUMP           : 0
Tier LOAD           : 0
```

Therefore, the formal F result should **not** be described as an SSD-offload speedup.

Instead, the cost-aware policy correctly determined that migration was not profitable under this workload and avoided the large I/O overhead observed in U.

F therefore preserves most of the Continuum benefit while avoiding harmful externalization.

---

## 9. KV Cache utilization

For the final Full System at concurrency 128:

```text
Mean KV utilization : 28.44%
P95                  : 61.03%
P99                  : 65.31%
Max                  : 71.69%
```

The peak value should not be interpreted as typical utilization.

The project was tested successfully at concurrency 128. This establishes stability **at least up to the tested C128 configuration**; it does not establish C128 as the system's absolute maximum capacity.

---

## 10. TTL metric interpretation

The reported `TTL hit` metric is an effective replay metric under the project's scaled / capped replay semantics.

It is **not** the raw prediction accuracy of the original EnvBench timing trace.

For the final formal runs:

```text
C TTL hit : ~44.18%
F TTL hit : ~46.38%
```

See `docs/BENCHMARK.md` for the exact replay semantics and metric definitions.

---

## 11. Workload construction

The repository contains generated benchmark workloads under:

```text
benchmark/workloads/
```

The formal workload was constructed by filtering complete trajectories such that every runtime event satisfies the serving input limit.

Final construction summary:

```text
Source eligible single-call trajectories : 3677
Accepted complete trajectories            : 1274
Rejected trajectories                     : 2403
Accepted tool events                       : 13419
Expected requests                          : 14693
Accepted input tokens                      : 24,822,191
```

Accepted input-token distribution:

```text
min    : 433
mean   : 1849.78
median : 1755
p90    : 3142
p95    : 3429
p99    : 3804
max    : 3997
```

The workload also implements reset-aware prefix segmentation for trajectories whose input-token sequence decreases across phases.

See:

- `benchmark/scripts/build_full_envbench_workload.py`
- `benchmark/scripts/build_full_envbench_shards.py`
- `docs/BENCHMARK.md`

---

## 12. Quick start

### 12.1 Clone the three repositories

```bash
git clone git@github.com:YEYVHAIOU/vllm-prefix-experiment.git
git clone git@github.com:YEYVHAIOU/vllm-continuum.git
git clone git@github.com:YEYVHAIOU/unified-cache-management-continuum.git
```

Keep them as sibling directories:

```text
workspace/
├── vllm-prefix-experiment/
├── vllm-continuum/
└── unified-cache-management-continuum/
```

### 12.2 Check out the frozen source versions

```bash
cd vllm-continuum
git checkout vllm-continuum-final-20260820

cd ../unified-cache-management-continuum
git checkout ucm-continuum-final-20260822
```

### 12.3 Set machine-specific paths

Before launching the system:

```bash
export VENV_PATH=/path/to/your/vllm-environment
export MODEL_PATH=/path/to/Qwen3-0.6B
```

Optional overrides:

```bash
export VLLM_REPO=/path/to/vllm-continuum
export UCM_REPO=/path/to/unified-cache-management-continuum
export CUDA_HOME=/usr/local/cuda-12.8
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

The default source layout expects the two source repositories to be sibling directories of this repository.

### 12.4 Start the final Full System

```bash
cd vllm-prefix-experiment
bash deployment/scripts/start_full.sh
```

Check status:

```bash
bash deployment/scripts/status.sh
```

Run the deployment smoke test:

```bash
bash deployment/scripts/smoke_test.sh
```

Stop:

```bash
bash deployment/scripts/stop.sh
```

For the complete environment and dependency procedure, read:

```text
docs/DEPLOYMENT.md
docs/COMPATIBILITY.md
```

---

## 13. Reproducing the V/U/C/F benchmark

The case configurations are:

```text
benchmark/configs/V.env
benchmark/configs/U.env
benchmark/configs/C.env
benchmark/configs/F.env
```

Common benchmark configuration:

```text
benchmark/configs/common.env
```

The main formal runner is:

```text
benchmark/scripts/run_full_envbench_vucf.sh
```

The run order and auxiliary commands are documented in:

```text
benchmark/README_RUN_ORDER.md
docs/BENCHMARK.md
```

Generated runtime outputs are intentionally ignored by Git and are written under runtime directories.

---

## 14. Deployment versus frozen release

This GitHub repository is a **curated source / reproduction repository**.

The original frozen research release additionally contained:

- Golden Runtime source snapshots,
- runtime logs,
- pooled request-level outputs,
- large integrity manifests,
- raw archival material,
- local Git bundles.

Those artifacts are intentionally excluded from the normal Git repository.

The curated GitHub repository should therefore not be interpreted as a byte-for-byte copy of the complete frozen release.

Historical paths such as:

```text
/root/autodl-tmp/...
deployment/runtime/golden/...
```

may still appear in provenance documents and frozen benchmark manifests because they record the environment used for the original validated run.

They are historical evidence, not required GitHub checkout paths.

---

## 15. Documentation

Detailed documentation:

- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) — installation, deployment, startup, validation, and recovery.
- [`docs/BENCHMARK.md`](docs/BENCHMARK.md) — workload construction, metrics, V/U/C/F protocol, and result interpretation.
- [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md) — compatibility modifications and integration details.
- [`docs/SOURCE_PROVENANCE.md`](docs/SOURCE_PROVENANCE.md) — source lineage, commits, tags, runtime snapshots, and verification.
- [`docs/min3500_capacity_C76_C96_C128_final_summary.md`](docs/min3500_capacity_C76_C96_C128_final_summary.md) — high-pressure capacity campaign summary.

---

## 16. Compatibility notes

The final system is **not** a simple unmodified combination of public upstream vLLM, Continuum, and UCM.

The working implementation required compatibility and integration modifications, including changes around:

- Continuum Dynamic TTL configuration and validation,
- scheduler / block-pool instrumentation,
- UCM's vLLM connector integration,
- DRAMStore / TieredStore compatibility,
- block-size and cache-size handling,
- transfer-policy control,
- request load behavior,
- empty-load initialization,
- final runtime and benchmark interfaces.

For the exact implementation history and compatibility details, see:

```text
docs/COMPATIBILITY.md
docs/SOURCE_PROVENANCE.md
```

---

## 17. Reproducibility and interpretation caveats

When using or citing these results, keep the following constraints explicit:

1. The benchmark is an EnvBench-derived **serving replay**, not semantic task-accuracy evaluation.
2. `TTL hit` is the project's effective replay metric, not raw EnvBench prediction accuracy.
3. The `min3500` pressure workload uses:

   ```text
   pressure_tokens = max(original_input_tokens, 3500)
   ```

   It does not mean every original natural input already contained at least 3500 tokens.
4. F's final formal result did not rely on successful SSD migration for its speedup; the cost-aware policy selected zero externalized blocks in that campaign.
5. C128 is the highest formally reported tested concurrency in the final campaign, not a proven absolute capacity limit.
6. Absolute performance numbers are hardware-, software-, model-, and workload-dependent.

---

## 18. Final release identity

```text
release_date             = 2026-08-23

model                    = Qwen/Qwen3-0.6B

formal_benchmark         = Full Runtime-Eligible EnvBench
formal_concurrency       = 128
formal_trajectories      = 1274
formal_tool_events       = 13419
formal_requests_per_case = 14693
formal_cases             = V,U,C,F
formal_case_runs         = 36
formal_total_requests    = 58772
formal_validation        = PASS
```

The canonical machine-readable identity is stored in:

```text
manifests/final_release_identity.txt
```

---

## 19. License and upstream attribution

This repository combines research artifacts that interact with multiple upstream projects.

Before redistributing a public release, review and preserve the applicable upstream license files, copyright notices, and dataset redistribution terms.

In particular:

- preserve upstream vLLM notices,
- preserve UCM notices and license terms,
- verify EnvBench redistribution terms before publishing raw source data,
- do not treat this repository as an official upstream distribution.

