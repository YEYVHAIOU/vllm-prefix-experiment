# EnvBench-derived `min3500` 高并发压力实验总结（C76 / C96 / C128）

**冻结日期：2026-08-22**  
**用途：作为当前阶段 Stress / Capacity Benchmark 的最终阶段性记录。后续 Full EnvBench 实验只做增量补充，不再回头重构本节。**

---

## 0. 一句话结论

在 RTX 4090 + Qwen3-0.6B、`max_model_len=4096`、GPU memory utilization 0.80 的固定环境下，EnvBench-derived `min3500` 高并发压力 workload 在 C76、C96、C128 三档上均完成了 V/U/C/F 四系统的 100% 请求成功验证。

当前最稳定、最清晰的结论是：

1. **Continuum 是当前性能收益的主体。** C/F 在高 KV 压力下将 Prefix Cache hit 提升到约 87%–90%，请求吞吐约为 V 的 3.2–3.6 倍。
2. **naive UCM（eager + full）是明显的负面对照。** 它几乎没有改善 GPU Prefix hit，却产生 26.48 / 33.49 / 44.69 GiB 的 SSD backing 和大量重复 load，吞吐显著低于 V。
3. **Full System（F）的 cost-aware UCM 在 C76/C96/C128 均选择 `selected=0`。** 因而当前压力测试证明的是“能识别不值得迁移并避免 naive offload 的灾难性开销”，而不是“已经通过 SSD offload 获得额外性能收益”。
4. **C128 全部 PASS，因此当前只能说“已验证稳定到并发 128”，不能说“最大容量就是 128”。**
5. 当前 workload 是 **pressure-enhanced EnvBench-derived stress workload**，不是完整 EnvBench 分布；完整 EnvBench 将作为下一阶段 Full-distribution Evaluation。

---

# 1. 实验目标

本阶段不是用来模拟 EnvBench 的真实总体分布，而是专门回答：

> 在长上下文、多轮工具调用、高并发、高 KV Cache 压力下，vLLM / UCM / Continuum / Full System 的稳定性、延迟、吞吐、Prefix Cache、Dynamic TTL 与外存迁移行为分别如何？

因此它属于：

**EnvBench-derived Long-context High-contention Stress / Capacity Benchmark**

而不是：

**Full EnvBench benchmark**

两者后续必须分开表述。

---

# 2. 固定运行环境

## 2.1 硬件

- GPU：NVIDIA GeForce RTX 4090
- GPU Memory：24564 MiB
- Driver：595.71.05
- CPU：16 cores（当前 AutoDL 实例）
- Host RAM：120 GB
- AutoDL 数据盘：100 GB（扩容后）
- 运行阶段使用本地数据盘作为 UCM SSD backing，不使用 `/dev/shm` 冒充 SSD。

## 2.2 软件环境

已确认的主环境：

- Python：3.12.3
- PyTorch：2.8.0+cu129
- PyTorch CUDA runtime：12.9
- CUDA Toolkit：12.8
- nvcc：12.8 / V12.8.93
- GCC：11.2
- vLLM runtime：0.1.dev10+g05f00f8a8
- `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`
- `PYTHONHASHSEED=123456`

> Transformers / Tokenizers / huggingface_hub 的最终文档版本仍应在最终部署文档 freeze 前再用当前 venv 打印一次；本节不冻结此前存在过差异的包版本。

## 2.3 模型与服务参数

- Model：`Qwen/Qwen3-0.6B`
- `max_model_len=4096`
- GPU memory utilization：0.80
- swap：1 GiB
- Prefix caching：开启
- vLLM `enforce_eager`：关闭（即 launcher 输出的 `eager: disabled`；这与 UCM 的 `WHEN=eager` 是两件不同的事）
- `max_output_tokens=8`
- metrics interval：0.2 s
- request timeout：300 s
- lane assignment：`balanced`
- HTTP mode：`httpclient_per_lane`

## 2.4 工具时序回放参数

Capacity / stress workload：

- `duration_scale=0.01`
- `max_sleep_seconds=2`

因此：

> 本阶段 TTL hit / JCT 等“涉及工具等待”的指标反映的是 **0.01× 缩放后的 runtime replay**，不能直接当成原始 EnvBench 的真实 wall-clock tool duration。

---

# 3. 源码身份

## vLLM + Continuum

- branch：`joint-offload-v1`
- commit：`6e7d571b831e6e4b82f1b2f8228cc84e9a0261a2`
- tag：`vllm-continuum-final-20260820`

## UCM

- branch：`joint-offload-v1`
- commit：`be59181f50e515496a7f19a38178c8c2a05cf251`
- tag：`ucm-continuum-final-20260822`

该 UCM commit 已包含 benchmark 过程中发现并修复的 empty-load / KV-cache initialization compatibility bug。

---

# 4. Workload 定义：`min3500` 到底是什么

## 4.1 必须使用的正式定义

当前数据集不能只描述成：

> “筛选出至少 3500 token 的 EnvBench 样本”。

更准确的是：

> **从 EnvBench-derived trajectory workload 构造的 trajectory-preserving high-contention pressure workload；对原始 input tokens 小于 3500 的事件进行 pressure padding，使 replay input 至少达到 3500 tokens，同时保留 `original_input_tokens` 与 `pressure_padding_tokens`。**

构造逻辑的核心是：

```text
pressure_input_tokens = max(original_input_tokens, 3500)
```

上限仍受当前 workload / serving 配置约束，约 4000 token。

## 4.2 C76 基础 workload

文件：

`envbench_high_contention_balanced_min3500.csv`

元数据：

- Tool events：1225
- Trajectories：76
- Terminal requests：76
- Expected total requests：1301
- Pressure enhanced：true

压力化后的 input token：

- min：3500
- mean：3505.0865
- median：3500
- max：3988

原始 input token：

- min：433
- mean：1625.1878
- median：1451
- max：3988

这意味着当前 stress workload 中大量短于 3500 的原始事件被补长，而不是被简单丢弃。

工具调用组成：

- `execute_bash_command`：10
- `get_directory_contents`：359
- `get_file_contents`：365
- `check_presence`：141
- `inspect_header`：263
- `submit_documentation`：87

原始 tool duration：

- min：0.004919 s
- mean：1.133644 s
- median：0.028315 s
- max：386.264591 s

运行时使用 `duration_scale=0.01` 和 `max_sleep_seconds=2`。

## 4.3 Capacity 扩展 workload

| 档位 | Trajectories | Tool events | Total requests |
|---|---:|---:|---:|
| C76 | 76 | 1225 | 1301 |
| C96 | 96 | 1540 | 1636 |
| C128 | 128 | 2079 | 2207 |

C96 / C128 与 C76 属于同一 `min3500` pressure workload family，用更大的 trajectory 集增加并发与整体 KV 压力。

**重要：C76 / C96 / C128 的 workload 组成不完全相同，因此跨档位的 req/s 不能被简单解释为“同一 workload 只改变 concurrency 的 scaling curve”。**  
跨档位更适合观察稳定性、KV 压力、趋势；同档 V/U/C/F 才是最严格的横向对比。

---

# 5. 四组系统定义

| Case | Scheduler | UCM | WHEN | WHAT | WHERE | 角色 |
|---|---|---|---|---|---|---|
| V | FCFS | OFF | — | — | GPU KV | vanilla vLLM baseline |
| U | FCFS | ON | eager | full | TieredStore, DRAM 128 MiB + SSD | naive / aggressive UCM ablation |
| C | Continuum | OFF | — | — | GPU KV | Continuum / Dynamic TTL |
| F | Continuum | ON | joint + cost_full | frontier_tail K=4 | TieredStore, DRAM 128 MiB + SSD | Full System |

### 两个容易混淆的点

1. **U 的 `WHEN=eager` 是 UCM migration policy。**
2. Launcher 的 `eager: disabled` 是 vLLM `enforce_eager` 没开，允许正常 CUDA Graph 路径。

### Frontier-Tail 的正式命名

当前 `frontier_tail K=4` 应称为：

**Sparse External KV Backing / Frontier-Tail Partial Retention**

不要写成 UCM 官方 Sparse Attention。当前实现是外部 KV backing 的部分保留策略，不等价于修改 attention 计算本身的 sparse attention。

---

# 6. 指标字典

## 6.1 Correctness / Stability

### Requests total / success / failed

- total：本轮实际发送请求数
- success：成功完成请求数
- failed：请求失败数
- success rate：`success / total`

Capacity 稳定判据：

- 100% expected requests success
- 无 OOM
- 无 EngineCore crash
- 无 server fatal
- 无 request timeout
- server / port lifecycle 能正常收尾

**P99 较高本身不等于“不稳定”。**

---

## 6.2 Throughput

### Request throughput

```text
successful requests / benchmark elapsed wall time
```

单位：req/s。

### Input / Output / Total token throughput

```text
input_tokens / elapsed
output_tokens / elapsed
(input_tokens + output_tokens) / elapsed
```

单位：tok/s。

Stress workload 输出固定很短（最多 8 tokens），因此总体算力/时延主要被长 prompt prefill、缓存命中与调度行为支配。

---

## 6.3 Latency

### TTFT — Time To First Token

从请求发出到首 token 返回。

在本项目中是最重要的 Prefix Cache / prefill 受益指标之一。

### TPOT — Time Per Output Token

首 token 之后的平均生成 token 间隔。

由于 output 很短，本阶段 TPOT 的重要性低于 TTFT。

### E2E

单次请求完整端到端延迟。

### 分位数

每项记录：

- Mean
- Median / P50
- P90
- P95
- P99
- Max

---

## 6.4 JCT — Job / Trajectory Completion Time

当前 summarizer 的正式定义：

对每个 trajectory/job：

```text
JCT = max(request_start + request_e2e) - min(request_start)
```

再对所有 trajectories 求 Mean / Median / P90 / P95 / P99 / Max。

JCT 比单请求 E2E 更符合多轮 Agent 的整体任务完成体验。

**注意：由于工具等待使用 0.01× duration scale，本阶段 JCT 是 pressure runtime replay JCT，不是原始 EnvBench 实际任务 wall time。**

---

## 6.5 GPU Prefix Cache

### Query tokens

被 prefix-cache lookup 查询的 token 总量。

### Hit tokens

成功由 GPU Prefix Cache 复用的 token 数。

### Miss tokens

```text
query_tokens - hit_tokens
```

### Hit rate

```text
hit_tokens / query_tokens
```

这是当前最直观的缓存复用效果指标。

---

## 6.6 Scheduler / Dynamic TTL

### Preemptions

vLLM / Continuum 调度中发生的 preemption 数量。

### TTL matched

工具事件成功匹配到服务端 Dynamic TTL 决策的数量。

### TTL hit

工具调用返回时间没有超过预测 TTL，即：

```text
observed/scaled tool gap <= TTL
```

### TTL timeout

工具调用等待超过 TTL。

### TTL hit rate

```text
TTL hit / TTL matched
```

### TTL history source

- `fixed_cold_start`：历史不足时使用默认/cold-start
- `global_history`：使用全局工具调用历史
- `per_tool_history`：该工具已有足够历史，使用 tool-specific history

后期绝大多数事件进入 `per_tool_history`，说明 estimator 已进入稳定在线历史模式。

---

# 7. KV Cache 指标含义

### `kv_cache_usage_perc`

vLLM 报告的 GPU KV Cache occupancy，0–1。

### `num_requests_running`

采样时正在 GPU scheduler 中运行的请求数。

### `num_requests_waiting`

等待调度的请求数。

### `continuum_pinned_requests`

被 Dynamic TTL / Continuum 保护的请求数。

### `continuum_pinned_blocks`

被 TTL 保护、当前不应被驱逐的 KV blocks。

### `continuum_running_blocks`

当前 running requests 使用的 KV blocks。

### `continuum_shared_pinned_running_blocks`

同时属于 pinned 与 running 的重叠 blocks。

### `continuum_active_blocks`

项目 instrumentation 中用于表示 active KV working set 的 blocks。

### `continuum_active_unpinned_blocks`

active 但不属于 pinned protection 的 blocks。

### `continuum_free_blocks`

缓存管理视角下的 free blocks。

### `continuum_true_free_blocks`

更严格意义上当前没有被 active/cache state 占用的 blocks；高压阶段该值接近 0 说明 GPU KV 空间基本被吃满。

### `continuum_evictable_cached_blocks`

仍缓存着历史 prefix，但当前不 pinned、不 active，可被回收的 blocks。

### `continuum_total_blocks`

本轮可用 KV block 总数，当前约 9530。

### `continuum_evicted_blocks_total`

累计 eviction counter。

**这是累计计数器，因此正式报告更适合使用 final/max，而不是把它的采样 Mean/P95 当作“典型驱逐量”。**

---

# 8. UCM 指标含义

## WHEN

### decisions

发生了多少次 offload decision。

### candidate

进入 WHEN 决策的候选 blocks。

### effective

通过前置有效性/资格筛选后真正值得参与 cost evaluation 的候选量。

### selected

最终决定 external backing/offload 的 blocks。

### skipped

未选择迁移的候选 blocks。

---

## WHAT

### candidate

进入保留/裁剪策略的 blocks。

### retained

最终保留并准备 external backing 的 blocks。

### pruned

被 WHAT 策略裁掉的 blocks。

`frontier_tail K=4` 只有在 WHEN 真正选择迁移之后才有实际迁移意义。

---

## WHERE / TieredStore

### Tier CREATE

本轮创建 external KV backing 时分配到：

- DRAM
- SSD

的 blocks。

### Tier DUMP

GPU → external tier 的 block 写出。

### Tier LOAD

external tier → GPU 的 block 读取。

**LOAD 是累计传输次数，可以远大于 unique stored blocks，因为同一 backing 可被多次访问。**

---

## SSD backing bytes / GiB

runner 在 case 完成时记录 `ssd_store` 的落盘大小，然后自动清理目录。

它表示 **run-end external backing footprint**；在当前没有在线 SSD eviction 的实现下，也可近似反映本轮最高存储规模，但正式表述优先使用“run-end backing footprint”，不要未经证明称作精确 peak。

---

# 9. 主结果总表

## 9.1 请求正确性

所有 12 个正式 case：

| Concurrency | V | U | C | F |
|---|---|---|---|---|
| C76 | 1301/1301 PASS | 1301/1301 PASS | 1301/1301 PASS | 1301/1301 PASS |
| C96 | 1636/1636 PASS | 1636/1636 PASS | 1636/1636 PASS | 1636/1636 PASS |
| C128 | 2207/2207 PASS | 2207/2207 PASS | 2207/2207 PASS | 2207/2207 PASS |

均为：

- failed = 0
- runtime errors = 0
- `WORKLOAD_RESULT=PASS`
- `CASE_RUN=PASS`

因此：

> **当前 stress workload 已验证 V/U/C/F 均能稳定运行到并发 128。**

不能写：

> “最大稳定并发是 128”。

因为没有测试 >128。

---

# 10. C76 结果

统一 token：

- Input tokens：4,561,154
- Output tokens：9,876
- Total tokens：4,571,030

## 10.1 主指标

| Metric | V | U | C | F |
|---|---:|---:|---:|---:|
| Request throughput req/s | 25.7354 | 9.9903 | **91.6935** | 82.9793 |
| Total token throughput tok/s | 90,420.62 | 35,100.76 | **322,162.81** | 291,545.54 |
| Prefix hit | 21.98% | 23.42% | **90.45%** | 89.59% |
| Preemptions | 0 | 0 | 14 | 18 |
| TTL hit | N/A | N/A | 97.71% | **98.37%** |
| SSD backing | 0 | **26.4767 GiB** | 0 | 348 B |

C 相对 V：

- request throughput：约 **3.56×**
- Prefix hit：+68.48 percentage points

F 相对 V：

- request throughput：约 **3.22×**
- Prefix hit：+67.61 pp

U 相对 V：

- request throughput 仅为 V 的约 **38.8%**
- Prefix hit 只增加约 **1.44 pp**

## 10.2 已冻结的详细 latency（当前 clean C76 记录）

### U

- TTFT：mean 4.990125 / median 4.027832 / P90 8.233718 / P95 15.532725 / P99 23.433323 / max 28.168334
- TPOT：mean 0.324697 / median 0.303368 / P90 0.561114 / P95 0.659786 / P99 0.711773 / max 0.786976
- E2E：mean 7.130238 / median 7.148395 / P90 10.232417 / P95 17.200386 / P99 24.812790 / max 29.113996
- JCT：mean 122.370536 / median 127.124726 / P90 128.854686 / P95 129.175726 / P99 129.874833 / max 130.129726

### C

- TTFT：mean 0.312498 / median 0.111301 / P90 0.184882 / P95 0.680258 / P99 7.445288 / max 8.980150
- TPOT：mean 0.035526 / median 0.038076 / P90 0.047987 / P95 0.050115 / P99 0.057583 / max 0.173168
- E2E：mean 0.546656 / median 0.379983 / P90 0.528781 / P95 0.891944 / P99 7.707035 / max 9.211415
- JCT：mean 9.587862 / median 9.835659 / P90 12.834528 / P95 13.211290 / P99 13.578083 / max 14.107480

### F

- TTFT：mean 0.400106 / median 0.184230 / P90 0.277774 / P95 0.698274 / P99 8.646604 / max 10.559174
- TPOT：mean 0.036099 / median 0.038141 / P90 0.050382 / P95 0.052569 / P99 0.058309 / max 0.083417
- E2E：mean 0.638045 / median 0.461610 / P90 0.602374 / P95 0.938070 / P99 8.880019 / max 10.798172

> C76 pilot 的最终 JCT summarizer 修复是在试验后完成的；当前冻结记录中 U/C 已重新展开 JCT，F/V 的完整 JCT 不作为本节核心对比。C96/C128 已对四组统一使用最终 JCT 逻辑。

## 10.3 C76 UCM

### U

- WHEN decisions：225
- candidate/effective/selected：15,565 / 15,565 / 15,565
- skipped：0
- WHAT：full，retained 15,565
- Tier create：73 DRAM + 15,492 SSD
- Tier dump：73 DRAM + 15,492 SSD
- Tier load：798 DRAM + 196,079 SSD
- SSD backing：26.476661 GiB

### F

- WHEN candidates：18,522
- effective：471
- selected：0
- skipped：18,522
- Tier create/dump/load：0
- SSD backing：348 B

---

# 11. C96 结果

统一：

- Requests：1636
- Input tokens：5,737,152
- Output tokens：12,416
- Total tokens：5,749,568

## 11.1 主指标

| Metric | V | U | C | F |
|---|---:|---:|---:|---:|
| Req/s | 26.0756 | 9.1828 | **87.6528** | 85.6337 |
| Total tok/s | 91,640.30 | 32,272.09 | **308,047.45** | 300,951.46 |
| TTFT mean | 3.0614 | 8.1440 | **0.4769** | 0.4842 |
| E2E mean | 3.2246 | 9.6718 | **0.7177** | 0.7237 |
| JCT mean | 55.1378 | 165.1558 | **12.5295** | 12.6224 |
| Prefix hit | 19.57% | 20.11% | 87.72% | **88.30%** |
| Preemptions | 0 | 0 | 19 | 27 |
| TTL hit | N/A | N/A | **98.44%** | 98.31% |
| SSD | 0 | **33.4889 GiB** | 0 | 348 B |

### 相对 V

C：

- Req throughput：**3.36×**
- TTFT mean：降低 **84.42%**
- JCT mean：降低 **77.28%**

F：

- Req throughput：**3.28×**
- TTFT mean：降低 **84.19%**
- JCT mean：降低 **77.11%**

U：

- Req throughput 仅为 V 的 **35.2%**
- 即吞吐降低 **64.78%**

### F vs C

- F throughput 比 C 低 **2.30%**
- Prefix hit 比 C 高约 **0.58 pp**
- TTFT/E2E/JCT 基本同一量级

由于本阶段每档不是重复统计实验，不应把 2.3% 当成严格统计显著差异；应称为“observed overhead / observed gap”。

---

## 11.2 C96 latency 全分位

### V

- TTFT：3.061380 / 3.807376 / 3.863280 / 3.871759 / 3.894843 / 4.138271
- TPOT：0.024770 / 0.026326 / 0.027123 / 0.027583 / 0.028759 / 0.038211
- E2E：3.224604 / 3.990925 / 4.047952 / 4.056567 / 4.077992 / 4.308831
- JCT：55.137839 / 57.802656 / 61.169295 / 61.639376 / 62.514095 / 62.636270

顺序均为：

`mean / median / P90 / P95 / P99 / max`

### U

- TTFT：8.143978 / 9.229160 / 10.280702 / 20.283509 / 29.077851 / 34.197219
- TPOT：0.231870 / 0.144210 / 0.627706 / 0.709264 / 0.794632 / 0.860346
- E2E：9.671831 / 10.254556 / 14.778803 / 21.821153 / 30.311548 / 35.163746
- JCT：165.155771 / 172.926982 / 176.562732 / 177.035988 / 177.896025 / 177.902834

### C

- TTFT：0.476861 / 0.159183 / 0.250711 / 0.812369 / 9.999997 / 14.026327
- TPOT：0.036542 / 0.039597 / 0.048465 / 0.050680 / 0.056968 / 0.098766
- E2E：0.717657 / 0.427408 / 0.563506 / 1.040210 / 10.228585 / 14.320482
- JCT：12.529458 / 13.734957 / 16.907675 / 17.466896 / 17.747790 / 18.536855

### F

- TTFT：0.484151 / 0.173546 / 0.290432 / 0.808254 / 9.996655 / 13.679775
- TPOT：0.036353 / 0.039590 / 0.051110 / 0.053306 / 0.058069 / 0.210618
- E2E：0.723696 / 0.452403 / 0.571791 / 1.099865 / 10.098926 / 14.043800
- JCT：12.622445 / 13.596322 / 17.563788 / 17.716661 / 18.416454 / 18.977418

---

## 11.3 C96 KV pressure

### C

`kv_cache_usage_perc`

- mean：0.739005
- median：0.949843
- P90：0.989885
- P95：0.991532
- P99：0.997242
- max：0.999370

`continuum_active_blocks`

- mean：7042.72
- median：9052
- P90：9433.6
- max：9524

`true_free_blocks`

- mean：662.14
- median：37
- max：8632

### F

KV usage：

- mean：0.743214
- median：0.947901
- P90：0.990304
- P95：0.992865
- P99：0.999104
- max：0.999265

active blocks：

- mean：7082.83
- median：9033.5
- P90：9437.6
- max：9523

true free：

- mean：758.54
- median：35
- max：9183

结论：

> C96 中 C/F 的 median KV occupancy 已约 95%，P95/P99 接近 100%，属于真实的 GPU KV 高压阶段。

---

## 11.4 C96 UCM

### U

- WHEN decisions：289
- candidate/effective/selected：19,668 / 19,668 / 19,668
- skipped：0
- WHAT retained：19,668
- Tier create：73 DRAM + 19,595 SSD
- Tier dump：73 DRAM + 19,595 SSD
- Tier load：798 DRAM + 258,983 SSD
- `UCM_TIER_LOOKUP`：SSD hits 300,623
- SSD backing：33.488911 GiB

### F

- WHEN decisions：336
- candidates：23,607
- effective：612
- selected：0
- skipped：23,607
- Tier create/dump/load：0
- reasons：
  - `joint_cost_full_skip`：245
  - `joint_cost_full_terminal_skip`：2
  - `joint_cost_full_zero_evict_skip`：89
- SSD backing：348 B

---

# 12. C128 结果

统一：

- Requests：2207
- Input tokens：7,738,685
- Output tokens：16,760
- Total tokens：7,755,445

## 12.1 主指标

| Metric | V | U | C | F |
|---|---:|---:|---:|---:|
| Req/s | 26.2482 | 10.7967 | **92.7126** | 86.0521 |
| Total tok/s | 92,236.59 | 37,939.68 | **325,793.91** | 302,388.85 |
| TTFT mean | 4.0807 | 9.3066 | **0.6275** | 0.6849 |
| E2E mean | 4.2434 | 10.7338 | **0.8687** | 0.9358 |
| JCT mean | 73.3681 | 185.4194 | **15.2992** | 16.4636 |
| Prefix hit | 17.80% | 17.84% | **87.61%** | 87.51% |
| Preemptions | 0 | 0 | 30 | 34 |
| TTL hit | N/A | N/A | **98.80%** | 98.56% |
| SSD | 0 | **44.6866 GiB** | 0 | 348 B |

### 相对 V

C：

- throughput：**3.53×**
- TTFT mean：降低 **84.62%**
- JCT mean：降低 **79.15%**

F：

- throughput：**3.28×**
- TTFT mean：降低 **83.22%**
- JCT mean：降低 **77.56%**

U：

- throughput 仅为 V 的 **41.13%**
- 即降低 **58.87%**

### F vs C

- F throughput 比 C 低约 **7.18%**
- Prefix hit 基本相同（87.51% vs 87.61%）
- JCT：16.46 s vs 15.30 s

仍然只能描述为该单轮 capacity run 的 observed gap。

---

## 12.2 C128 latency 全分位

### V

- TTFT：4.080748 / 5.054718 / 5.116564 / 5.130531 / 5.163129 / 5.520294
- TPOT：0.024661 / 0.025844 / 0.026780 / 0.027520 / 0.028437 / 0.037019
- E2E：4.243372 / 5.235282 / 5.297040 / 5.312292 / 5.344913 / 5.701913
- JCT：73.368107 / 75.380577 / 82.512959 / 83.059800 / 83.701723 / 83.861948

### U

- TTFT：9.306569 / 8.298527 / 12.841642 / 24.283640 / 34.943067 / 40.850173
- TPOT：0.216435 / 0.136321 / 0.527202 / 0.567441 / 0.630363 / 0.751152
- E2E：10.733757 / 11.773979 / 13.789453 / 25.383171 / 36.401191 / 41.780387
- JCT：185.419359 / 190.849801 / 202.694991 / 203.218656 / 203.999636 / 204.279706

### C

- TTFT：0.627544 / 0.163111 / 0.242300 / 1.056183 / 15.357464 / 18.406934
- TPOT：0.036571 / 0.037884 / 0.047918 / 0.050120 / 0.053472 / 0.103100
- E2E：0.868699 / 0.423336 / 0.556959 / 1.295467 / 15.667039 / 18.714256
- JCT：15.299203 / 15.727914 / 21.588191 / 22.298745 / 23.105126 / 23.686647

### F

- TTFT：0.684926 / 0.200961 / 0.297452 / 1.064173 / 16.906547 / 20.236010
- TPOT：0.038042 / 0.040359 / 0.051084 / 0.055191 / 0.059981 / 0.389554
- E2E：0.935782 / 0.485707 / 0.601627 / 1.306957 / 17.098347 / 20.497838
- JCT：16.463575 / 17.417416 / 24.126525 / 24.865806 / 25.309658 / 25.476722

---

## 12.3 C128 KV pressure

### C

KV usage：

- mean：0.788710
- median：0.964690
- P90：0.989087
- P95：0.990005
- P99：0.991616
- max：0.991815

active blocks：

- mean：7516.41
- median：9193.5
- P90：9426
- max：9452

true free：

- mean：597.06
- median：32
- max：9145

### F

KV usage：

- mean：0.805749
- median：0.960441
- P90：0.988552
- P95：0.989386
- P99：0.991493
- max：0.992445

active blocks：

- mean：7678.78
- median：9153
- P90：9420.9
- max：9458

true free：

- mean：577.83
- median：29.5
- max：9145

结论：

> C128 中 C/F 的 median KV occupancy 约 96%，`true_free_blocks` median 只有约 30 blocks；系统长期运行在接近 KV 满载状态，但仍保持 100% 请求成功。

---

## 12.4 C128 UCM

### U

- WHEN decisions：386
- candidate/effective/selected：26,220 / 26,220 / 26,220
- skipped：0
- WHAT retained：26,220
- Tier create：73 DRAM + 26,147 SSD
- Tier dump：73 DRAM + 26,147 SSD
- Tier load：969 DRAM + 361,797 SSD
- `UCM_TIER_LOOKUP` SSD hits：420,769
- SSD backing：44.686631 GiB

### F

- WHEN decisions：439
- candidates：30,319
- effective：786
- selected：0
- skipped：30,319
- Tier create/dump/load：0
- reasons：
  - `joint_cost_full_skip`：350
  - `joint_cost_full_terminal_skip`：3
  - `joint_cost_full_zero_evict_skip`：86
- SSD backing：348 B

---

# 13. 横向机制分析

## 13.1 V：FCFS baseline 的瓶颈是等待，而不是 KV occupancy

V 在 C96/C128 的平均 KV usage 只有约 10%，看起来“不高压”，但这不能理解成 workload 没有压力。

原因是 FCFS baseline 同时运行的 request 数很少，大量 workload 排在 waiting queue：

### C96 V

- running mean：5.43
- waiting mean：76.99
- waiting median：89

### C128 V

- running mean：5.40
- waiting mean：104.24
- waiting median：121

所以：

> V 通过“让大量请求等待”避免 GPU KV 长时间满载，代价是 TTFT / JCT 很高。

这正是为什么不能只看 `kv_cache_usage_perc` 判断系统压力。

---

## 13.2 U：naive external KV migration 基本失败

U 的 Prefix hit 与 V 几乎一样：

| C | V hit | U hit | 增量 |
|---|---:|---:|---:|
| 76 | 21.98% | 23.42% | +1.44 pp |
| 96 | 19.57% | 20.11% | +0.54 pp |
| 128 | 17.80% | 17.84% | +0.04 pp |

但 U 的吞吐远低于 V，同时产生巨大的 external KV I/O。

SSD backing：

- C76：26.4767 GiB
- C96：33.4889 GiB
- C128：44.6866 GiB

按 trajectory 数归一化：

- C76：0.3484 GiB / trajectory
- C96：0.3488 GiB / trajectory
- C128：0.3491 GiB / trajectory

几乎线性。

这说明 aggressive `eager + full` 会近似按工作集规模持续把大量 KV 外迁。

更关键的是重复读取：

| C | SSD dump blocks | SSD load blocks | load / dump |
|---|---:|---:|---:|
| 76 | 15,492 | 196,079 | 12.66× |
| 96 | 19,595 | 258,983 | 13.22× |
| 128 | 26,147 | 361,797 | 13.84× |

因此 U 慢的核心证据链是：

```text
eager 选中几乎所有候选
→ full backing
→ 大量 KV 写入 SSD
→ 后续相同 backing 被反复 load
→ external transfer 开销远大于 Prefix reuse 收益
→ TTFT / JCT / throughput 全面恶化
```

正式结论：

> **External KV offload 不是越多越好。没有 cost-aware selection 的积极迁移可能产生巨大的数据搬运与存储开销，却无法形成有效的 GPU Prefix Cache 性能收益。**

---

## 13.3 C：当前性能提升的主要来源

C 在三档都没有 external UCM，但通过 Continuum scheduler + Dynamic TTL：

- C76 Prefix hit：90.45%
- C96：87.72%
- C128：87.61%

相比 V 的约 18%–22% 有数量级上的提升。

同时：

- C96 median KV occupancy：94.98%
- C128 median KV occupancy：96.47%

说明它不是简单减少并发，而是：

> 主动保持多轮 Agent 的有价值 prefix residency，并用接近满载的 GPU KV working set 换取高 reuse 与低 TTFT。

因此当前阶段必须明确：

> **Continuum 是 V→C/F 性能收益的主体。**

---

## 13.4 Dynamic TTL 的结果

C/F 的 TTL matching 均达到完整 tool event 覆盖：

### C96

- C：1540/1540，hit 98.44%
- F：1540/1540，hit 98.31%

### C128

- C：2079/2079，hit 98.80%
- F：2079/2079，hit 98.56%

C128 history source：

C：

- fixed cold start：8
- global history：33
- per-tool history：2038

F：

- fixed cold start：17
- global history：40
- per-tool history：2022

绝大多数决策已经进入 per-tool history。

因此可以说：

> 在当前 0.01× EnvBench runtime replay 下，Dynamic TTL 已经能稳定覆盖绝大多数 tool gap，并在压力升高到 C128 后仍维持约 98.5%–98.8% TTL hit。

不能直接说：

> “对原始 EnvBench tool duration 的预测准确率达到 98%”。

因为当前 duration 被缩放。

---

## 13.5 F：当前证明的是“避免错误迁移”，不是“offload 加速”

F 在三档：

- C76 selected = 0
- C96 selected = 0
- C128 selected = 0

即使 C96/C128 GPU KV median occupancy 已达到约 95%/96%，`cost_full` 仍认为 external migration 不划算。

因此 F 的实际外部迁移：

- Tier CREATE = 0
- Tier DUMP = 0
- Tier LOAD = 0
- SSD backing ≈ 348 B（目录/元数据量级）

这必须如实解释。

可以说：

> `joint + cost_full` 对候选块进行收益/成本判断，在当前 pressure workload 下成功避免了 naive UCM 的大规模无效迁移，使 Full System 保持接近 Continuum-only 的性能。

不能说：

> “F 依靠 SSD offload 在 C128 比 C 更快”。

事实上当前 C 的 raw throughput 还略高于 F。

---

# 14. F 与 C 的 observed overhead

F 相对 C 的请求吞吐：

- C76：约 -9.50%
- C96：约 -2.30%
- C128：约 -7.18%

但 Prefix hit 和 TTL hit 基本接近。

因为 F 没有真正发生 external KV transfer，这些差距可能来自：

- UCM decision path
- connector / metadata / lookup overhead
- logging / instrumentation
- 单轮运行噪声与 workload composition

当前 capacity 实验每档不是专门做统计显著性分析的多重复实验，因此正式表述应是：

> “Full System 在当前三档压力测试中保持与 Continuum-only 同一性能量级，并观察到约 2%–10% 的吞吐差距；由于 UCM 未触发实际 external transfer，该差距主要代表集成/决策路径开销，而不是 SSD I/O。”

不要把它包装成 F 显著优于 C。

---

# 15. 尾部延迟现象

C/F 的 Median、P90 通常非常低，但 P99 会出现秒级长尾，例如：

### C128 C

- TTFT median：0.163 s
- P90：0.242 s
- P95：1.056 s
- P99：15.357 s

### C128 F

- median：0.201 s
- P90：0.297 s
- P95：1.064 s
- P99：16.907 s

这说明：

> 大多数请求获得高 Prefix reuse 和低 TTFT，但极少部分请求在接近 KV 满载、preemption / queue interaction 条件下仍存在显著长尾。

因此最终报告不能只给 Mean，至少应保留 Median/P95/P99。

---

# 16. Capacity / Stability 冻结结论

当前已测试：

- C76
- C96
- C128

四 case 全 PASS。

因此正式措辞：

> **Under the EnvBench-derived min3500 long-context high-contention stress workload, all four configurations completed the expected requests without OOM, EngineCore crash, timeout, or server failure through concurrency 128. The evaluated stable capacity is therefore at least 128 concurrent trajectories under this benchmark configuration.**

中文：

> 在当前 EnvBench-derived `min3500` 长上下文高并发压力 workload 下，V/U/C/F 四组在 C76、C96、C128 均完成全部预期请求，未出现 OOM、EngineCore crash、请求失败或运行时错误。因此当前实验能证明“稳定运行能力至少达到并发 128”，但尚未测定真正的最大并发上限。

---

# 17. 本阶段最值得保留的数字

如果 PPT / 汇报只能放少量数据，优先保留 C128：

| 指标 | V | U | C | F |
|---|---:|---:|---:|---:|
| Req/s | 26.25 | 10.80 | **92.71** | 86.05 |
| TTFT Mean | 4.081 s | 9.307 s | **0.628 s** | 0.685 s |
| JCT Mean | 73.37 s | 185.42 s | **15.30 s** | 16.46 s |
| Prefix Hit | 17.80% | 17.84% | **87.61%** | 87.51% |
| TTL Hit | — | — | **98.80%** | 98.56% |
| SSD Backing | — | 44.69 GiB | — | ≈0 |

一句话：

> C128 下 Continuum-only / Full System 将 Prefix Cache hit 从约 18% 提升到约 88%，请求吞吐提升到 vanilla vLLM 的约 3.3–3.5 倍；相比之下 naive UCM 产生 44.69 GiB SSD backing 却几乎不提升 Prefix hit，并显著降低吞吐。

---

# 18. 不能夸大的结论

以下说法目前 **不能** 写：

### 不能 1

“Full System 的 SSD offload 提升了 C128 性能。”

原因：

`selected=0`，没有实际 Tier dump/load。

### 不能 2

“Frontier-Tail K=4 已在 C128 证明比 full retention 更快。”

原因：

F 没有选中 offload，WHAT 没有真正处理非零迁移候选。

### 不能 3

“最大稳定并发是 128。”

只能说：

“tested stable concurrency ≥128”。

### 不能 4

“这是完整 EnvBench 结果。”

这是 pressure-enhanced stress workload。

### 不能 5

“98% TTL hit 是原始 EnvBench tool-duration 预测准确率。”

当前 runtime tool gap 使用 0.01× scale。

### 不能 6

“UCM 本身没有价值。”

当前 U 证明的是 **naive eager/full offload** 在该 workload 上很差；不能推广成所有 UCM / external KV policy 都无价值。

---

# 19. 当前能够成立的研究叙事

推荐最终按下面逻辑陈述：

### 1. 问题

多用户 Agent serving 中，多轮请求被工具调用打断，理论上可复用的长 prefix 会与其他用户竞争有限 GPU KV，导致 Prefix Cache 被驱逐。

### 2. Continuum

Dynamic TTL + scheduler 根据 tool-return timing / history 判断 KV residency，使同一 Agent 的后续请求更容易命中此前 prefix。

### 3. naive UCM 反例

简单将大量 KV eager/full externalize 会产生巨大的 DRAM/SSD 搬运开销；仅仅“有二级缓存”并不等价于性能更好。

### 4. Full System

Continuum 的 temporal information 与 UCM 的 cost-aware WHEN 联合：

```text
WHEN: joint + cost_full
WHAT: frontier_tail K=4
WHERE: Tiered DRAM + SSD
```

目标不是“尽可能迁移”，而是：

> **only migrate when expected reuse value exceeds transfer / reload cost.**

### 5. 当前实验结果

在 C76/C96/C128 上 cost_full 全部选择 skip，说明当前 workload 下 recompute / GPU-local strategy 仍优于 externalization；系统避免了 U 的灾难性 SSD traffic，同时保留 Continuum 的主体收益。

### 6. 下一阶段真正要验证的点

完整 EnvBench 分布更杂，可能出现：

- 更长 idle tool gaps
- 不同 prompt lengths
- 不同 trajectory lengths
- 不同 reuse patterns

因此 Full EnvBench 的重点不是再次证明“能跑”，而是观察：

1. Continuum 的收益是否跨分布保持；
2. F 是否在部分更适合 externalization 的样本上出现 `selected > 0`；
3. selected > 0 时 Frontier-Tail + TieredStore 是否能以远低于 U 的存储/I/O量实现收益。

---

# 20. Evidence / Provenance

## C76

Manifest：

`/root/autodl-tmp/vllm_deploy_clean_20260820/benchmark/results/c76_pilot_20260822_210904/manifest.tsv`

Runs：

- V：`.../benchmark/runtime/runs/V_20260822_210904`
- U：`.../benchmark/runtime/runs/U_20260822_211043`
- C：`.../benchmark/runtime/runs/C_20260822_211344`
- F：`.../benchmark/runtime/runs/F_20260822_211446`

## C96

Manifest：

`/root/autodl-tmp/vllm_deploy_clean_20260820/benchmark/results/capacity_vucf_20260822_215703/manifest.tsv`

Runs：

- V：`V_20260822_215704`
- U：`U_20260822_215852`
- C：`C_20260822_220243`
- F：`F_20260822_220347`

## C128

Manifest：

`/root/autodl-tmp/vllm_deploy_clean_20260820/benchmark/results/capacity_vucf_20260822_220822/manifest.tsv`

Runs：

- V：`V_20260822_220822`
- U：`U_20260822_221031`
- C：`C_20260822_221450`
- F：`F_20260822_221600`

每个正式 run 内已保存：

- `summary.json`
- `requests.csv`
- `kv_cache_samples.csv`
- metrics before / after
- `server.log`
- `case_summary.json`
- `case_summary.txt`
- `ssd_store_bytes.txt`（U/F）
- correctness / runtime error evidence

runner 在记录 `ssd_store_bytes.txt` 后删除大型 SSD backing，避免 benchmark 结果目录无限膨胀。

---

# 21. 阶段冻结

从本文件开始：

**`min3500 + C76/C96/C128` 视为已完成的 Stress / Capacity Evaluation。**

后续不再修改其定位，而是在此基础上新增：

**Full EnvBench / Full-distribution Evaluation**

Full EnvBench 应作为新的实验章节，不覆盖本节数据。

最终实验结构建议固定为：

```text
A. Deployment Validation
B. EnvBench-derived Long-context High-contention Stress Benchmark
   - C76
   - C96
   - C128
   - V/U/C/F
C. Full EnvBench Distribution Benchmark
   - V/U/C/F
D. Distributional / Mechanism Analysis
   - context-length bins
   - trajectory-length bins
   - tool-duration bins
   - TTL source/hit
   - UCM selected/skipped
   - TieredStore I/O / footprint
```

---

## 最终冻结口径

> 当前压力实验已经证明：在长上下文、多轮 Agent、高并发且 GPU KV 接近满载的情况下，Continuum 的 tool-aware scheduling / Dynamic TTL 能显著提高 prefix reuse，并使请求吞吐和任务完成时间明显优于 vanilla vLLM。单独使用 naive eager/full UCM 会造成大量 SSD dump/load，几乎不改善 GPU prefix hit，反而严重拖慢系统；Full System 的 joint cost-aware policy 则在 C76–C128 均判断当前候选不值得外迁，避免了这类无效数据搬运，并保持接近 Continuum-only 的性能。该结果证明了 cost-aware migration selection 的必要性，但尚未证明 Frontier-Tail/SSD externalization 本身带来正向性能收益；后者留待更复杂的 Full EnvBench 分布进一步验证。
