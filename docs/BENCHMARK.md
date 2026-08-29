# 基准测试方法与结果

本文档记录 vLLM + Continuum + UCM 联合系统的最终评测方法、工作负载定义、指标聚合方式与正式实验结果。四种对比配置共享同一套最终代码基座、模型、硬件和服务参数，系统差异主要由调度策略与 UCM 功能开关产生。

正式主评测采用 **EnvBench 派生的服务系统回放**。实验关注吞吐、延迟、Prefix Cache 复用、GPU KV Cache 占用、Dynamic TTL 行为以及 KV 外部迁移，不评估 EnvBench 的语义任务正确率。

## 实验设计

最终实验比较 V、U、C、F 四种配置。四组使用相同模型、GPU、服务参数和工作负载，以尽量隔离调度策略与 KV 管理机制带来的影响。

| 配置 | 调度器 | Continuum | UCM | 主要机制 |
|---|---|---:|---:|---|
| **V** | FCFS | 关闭 | 关闭 | vLLM 基线 |
| **U** | FCFS | 关闭 | 开启 | eager WHEN + full WHAT + TieredStore |
| **C** | Continuum | 开启 | 关闭 | Dynamic TTL |
| **F** | Continuum | 开启 | 开启 | `cost_full` WHEN + Frontier-Tail WHAT + TieredStore |

完整系统 F 的策略组合为：

```text
Dynamic TTL
    +
cost_full WHEN
    +
Frontier-Tail WHAT (K=4)
    +
TieredStore WHERE (DRAM + SSD)
```

`Frontier-Tail` 是本项目实现的部分 KV 外部保留策略，用于控制跨 HBM 与外部存储层保留的 KV block 数量。它不改变 Attention 的数学定义，也不等价于 UCM 上游提供的 Sparse Attention 算法。

## 测试环境

正式实验运行于单卡 RTX 4090 环境。`max_model_len=4096` 是本项目服务实验采用的服务端限制，不代表 Qwen3-0.6B 的固有最大上下文长度。

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090，24564 MiB |
| CPU | 16 vCPU |
| 主机内存 | 约 120 GB |
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu129 |
| CUDA Toolkit | 12.8 |
| GCC | 11.2 |
| vLLM 运行时版本 | `0.1.dev10+g05f00f8a8` |
| 模型 | Qwen/Qwen3-0.6B |
| 服务端最大序列长度 | 4096 |
| 基准测试最大输入 | 4000 tokens |
| 最大输出 | 8 tokens |
| GPU 显存利用率参数 | 0.80 |
| Swap | 1 GiB |
| 并发度 | 128 |
| HTTP 模式 | `httpclient_per_lane` |
| 轨迹分配 | balanced |
| 工具等待时间缩放 | 0.01 |
| 最大回放等待时间 | 2 s |
| 上下文重置感知 | 开启 |

源码提交、标签与完整环境快照分别记录于 `docs/SOURCE_PROVENANCE.md` 和 `manifests/`。

## 工作负载与回放语义

### 正式 EnvBench 派生工作负载

正式工作负载文件为：

```text
benchmark/workloads/envbench_full_runtime_eligible_max4000.csv
```

对应审计文件：

```text
benchmark/workloads/envbench_full_runtime_eligible_max4000.audit.json
```

构造过程以规范化后的 EnvBench 可运行单调用轨迹为基础，并按完整 trajectory 进行筛选。只有当一个 trajectory 中所有运行事件都具有有效工具等待时间，且满足 `1 <= input_tokens <= 4000` 时，该 trajectory 才会进入正式工作负载；如果任一事件超过限制，则整个 trajectory 被排除。正式工作负载不截断 prompt，不进行 `min3500` 补长，也不使用高竞争评分或随机采样。

| 项目 | 数值 |
|---|---:|
| 源轨迹数 | 3677 |
| 源事件数 | 54078 |
| 接收轨迹数 | 1274 |
| 接收工具事件数 | 13419 |
| 每种配置请求数 | 14693 |
| 排除轨迹数 | 2403 |
| 接收事件输入 tokens | 24,822,191 |

接收输入 token 分布如下：

| 统计量 | Tokens |
|---|---:|
| Min | 433 |
| Mean | 1849.78 |
| Median | 1755 |
| P90 | 3142 |
| P95 | 3429 |
| P99 | 3804 |
| Max | 3997 |

### 确定性 token 前缀回放

正式基准测试不重建 EnvBench 原始 prompt 文本，而是根据记录的输入 token 长度生成确定性的 token ID 前缀序列。这样可以在不依赖模型语义输出的情况下，稳定重放 trajectory 中的前缀增长、工具调用顺序与等待关系，并直接观察服务系统中的缓存复用行为。

回放保留 trajectory 结构、工具调用顺序、工具等待时间、输入 token 长度、前缀增长关系以及阶段切换。它不测量代码生成质量、原始任务正确率或 EnvBench 最终任务成功率，因此本文结果属于服务系统评测，而不是语义任务评测。

### 上下文重置感知

1274 个接收轨迹中，622 个轨迹的输入 token 长度单调增长，652 个包含一次长度下降。原始轨迹检查表明，这类下降主要对应 search → build 的阶段切换。

当长度下降时，回放脚本会开启新的确定性 token 序列并增加 `context_segment`，避免把新的 build 阶段 prompt 错误解释为旧 search 阶段 prompt 的前缀。正式实验记录：

```text
reset_aware_prefix = true
context_reset_policy = new deterministic token stream after input-token decrease
```

### `min3500` 高压力工作负载

项目还保留一套独立的压力与并发承载实验工作负载：

```text
benchmark/workloads/envbench_high_contention_balanced_min3500.csv
```

其服务输入长度使用：

```text
pressure_tokens = max(original_input_tokens, 3500)
```

该工作负载用于 C76 / C96 / C128 的长上下文高压力实验，目的是放大 GPU KV Cache 竞争并验证系统稳定性。它不是正式 EnvBench 派生工作负载的自然长度分布，也不意味着原始 prompt 天然都至少包含 3500 tokens。完整结果见 `docs/min3500_capacity_C76_C96_C128_final_summary.md`。

## 指标与聚合方法

正式工作负载按完整 trajectory 确定性切分为 9 个分片，每个分片依次运行 V → U → C → F。最终 36 个运行实例全部通过正确性检查。

```text
V requests=14693 success=14693 failed=0
U requests=14693 success=14693 failed=0
C requests=14693 success=14693 failed=0
F requests=14693 success=14693 failed=0

cases=36
total_requests_all_cases=58772
FULL_36_CASE_VALIDATION=PASS
```

聚合逻辑由 `benchmark/scripts/aggregate_full_envbench.py` 实现。请求吞吐和 token 吞吐按所有分片的成功工作量除以工作负载回放耗时总和计算，不包含服务启动时间。TTFT、TPOT 与 E2E 的统计量从请求级样本重新汇总，而不是对各分片分位数求平均。

JCT 按 `shard + job_id` 重建。对同一 trajectory，JCT 定义为首个请求开始到最后一个请求完成之间的时间，再对 1274 个 trajectory 汇总分布。Prefix Cache 命中率由所有分片的 hit tokens 与 query tokens 累加后计算。

KV Cache 利用率来自周期采样的 `kv_cache_samples.csv`。TTL 计数按各运行摘要累加，TTL seconds 从 `tool_events.csv` 汇总。SSD 占用记录的是每个分片结束时的外部 KV 存储大小；不同分片顺序执行，并在每个配置运行结束后清理对应存储，因此这些数值按分片独立解释。

## 正式结果

### 整体性能

| 配置 | Req/s | Total tok/s | Prefix Cache 命中率 | Mean TTFT | Mean E2E | Mean JCT |
|---|---:|---:|---:|---:|---:|---:|
| **V** | 79.96 | 148,670.09 | 65.17% | 0.844 s | 0.976 s | 11.467 s |
| **U** | 25.86 | 48,074.55 | 67.16% | 2.793 s | 3.808 s | 44.142 s |
| **C** | **116.72** | **217,019.61** | **86.90%** | **0.302 s** | **0.490 s** | **5.873 s** |
| **F** | 109.24 | 203,122.35 | 86.89% | 0.327 s | 0.529 s | 6.325 s |

### 尾延迟

| 配置 | TTFT P50 | TTFT P95 | TTFT P99 | E2E P50 | E2E P95 | E2E P99 | JCT P50 | JCT P95 | JCT P99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V | 0.413 | 2.162 | 2.265 | 0.603 | 2.328 | 2.437 | 12.657 | 18.038 | 20.269 |
| U | 0.510 | 8.609 | 11.203 | 1.279 | 10.483 | 13.219 | 49.750 | 59.663 | 63.299 |
| C | 0.108 | 1.423 | 4.990 | 0.360 | 1.678 | 5.301 | 5.701 | 11.539 | 13.540 |
| F | 0.119 | 1.429 | 5.396 | 0.394 | 1.723 | 5.718 | 6.110 | 12.584 | 14.572 |

### GPU KV Cache 利用率

| 配置 | Mean | P95 | P99 | Max |
|---|---:|---:|---:|---:|
| V | 10.21% | 31.44% | 50.91% | 62.35% |
| U | 25.20% | 54.08% | 66.21% | 77.10% |
| C | 28.05% | 62.29% | 67.49% | 71.14% |
| F | **28.44%** | 61.03% | 65.31% | **71.69%** |

完整系统 F 在并发 128 下的平均 GPU KV Cache 利用率为 28.44%，P95 为 61.03%，P99 为 65.31%，峰值为 71.69%。这些统计来自整个正式工作负载的周期采样，峰值不代表典型运行状态。

### Dynamic TTL

| 配置 | TTL 命中率 |
|---|---:|
| V | N/A |
| U | N/A |
| C | 44.18% |
| F | 46.38% |

这里的 TTL 命中率比较预测 TTL 与实际回放中经过时间缩放和等待上限处理后的有效工具等待时间。它用于描述当前服务回放中的 TTL 命中行为，不等价于对原始 EnvBench 工具执行时间的预测准确率。

### UCM 与 SSD

U 配置在正式 9 分片实验中累计选择约 242,351 个 KV blocks。其分片结束时的 SSD 占用平均约 38.17 GiB，最大约 40.71 GiB。大量 KV 外部写入与读取同时伴随明显的吞吐和延迟退化。

F 配置使用 `joint + cost_full`。正式工作负载中 `when_selected=0`，Tier CREATE / DUMP / LOAD 均为 0，SSD 仅保留元数据量级的内容。这说明 F 的正式结果主要体现成本感知决策对收益不足迁移的过滤，而不是 SSD 迁移本身产生的加速。

## 结果分析

### Continuum 是主要性能收益来源

C 相比 V，将请求吞吐从 79.96 req/s 提升到 116.72 req/s，增幅约 45.97%；Prefix Cache 命中率从 65.17% 提升到 86.90%。同时平均 TTFT、E2E 与 JCT 分别下降约 64.18%、49.78% 和 48.78%。

这与项目关注的多轮 Agent 服务场景一致：同一 Agent 的前后轮请求被工具调用分隔后，上一轮 KV 是否仍能保留到下一轮，直接影响后续 Prefix Cache 复用与 Prefill 开销。Continuum / Dynamic TTL 在该工作负载中显著提高了可复用前缀在 GPU 中的驻留概率。

### eager/full 全量外迁在该工作负载中代价较高

U 与 V 使用相同 FCFS 调度器，但开启 eager/full UCM 外迁。Prefix Cache 命中率仅从 65.17% 增长到 67.16%，请求吞吐却下降到 25.86 req/s，平均 JCT 增长到 44.142 s。

结合 9 个分片中的大量已选择 KV blocks、SSD 存储占用与数据传输活动，可以看到外部存储容量本身并不会自动转化为性能收益。对于当前模型、硬件与工作负载，如果迁移和重新加载的代价高于未来重算或 GPU 本地复用的收益，积极外迁会成为额外的服务开销。

### 完整系统主要体现成本感知过滤

F 的 Prefix Cache 命中率与 C 基本一致，请求吞吐为 109.24 req/s，比 V 高 36.63%，但比 C 低 6.40%。由于正式实验中 F 没有选择实际 KV 外部迁移，这一差异主要反映联合系统额外的决策与 KVConnector 路径开销，而不是 SSD I/O。

当前完整系统的实验结果表明：Continuum 提供主要的前缀驻留性能收益，而 UCM 的成本感知策略能够避免 eager/full 全量外迁在该工作负载中产生的大规模无效外部 KV I/O。Frontier-Tail 与 TieredStore 已完成系统集成，但本次正式工作负载没有触发可用于独立评估其正向外部 I/O 收益的实际迁移。

## 压力与并发承载补充实验

除正式工作负载外，项目在 `min3500` 高压力工作负载上测试了 C76、C96 与 C128。四种配置在三个并发档位均完成全部预期请求，未出现 OOM、EngineCore crash、请求超时或服务端致命错误。

这一结果说明，在当前 RTX 4090、Qwen3-0.6B、`gpu_memory_utilization=0.80` 与对应高压力工作负载下，系统已经完成并发 128 的稳定性验证。更高并发未纳入本次实验，因此本文只描述已测试范围内的稳定性。

详细压力、KV Cache 占用、SSD I/O 与 C76/C96/C128 分档数据见：

```text
docs/min3500_capacity_C76_C96_C128_final_summary.md
```

## Open-loop 请求率饱和补充实验

为补充固定并发回放，本项目进一步使用 global ready queue 和 request-rate control 进行 open-loop 饱和实验，同时保持各 trajectory 的多轮因果关系。V、C、F 均完成 target 40、45、50、55、60、80 req/s 六个有效测试点，这些点均未触发客户端 `max_inflight=512` safety fuse。

在当前模型、RTX 4090 和 Full Runtime-Eligible workload 下，V 的 TTFT/E2E tail-latency knee 出现在约 48–53 actual req/s，约 57 actual req/s 后开始出现持续积压。C 与 F 的 steady throughput 饱和区间与 V 接近，但 P95/P99 tail latency 在约 44–48 actual req/s 已明显抬升；同时高负载下二者的 P50 仍可低于 V，表明 priority-aware scheduling 主要改变了延迟分布，而不能简单概括为整体服务能力下降。

F 的所有正式 saturation 点均未观察到实际 SSD-backed KV migration。U 的 eager/full saturation 尝试则在 target 15 req/s 时先因 SSD backing 数据耗尽 100 GB 实验盘，因此没有得到可用于容量判断的 U saturation 曲线。

完整方法、数据表、解释边界和结果图见 [`SATURATION_OPEN_LOOP_20260825.md`](SATURATION_OPEN_LOOP_20260825.md)。

## 实验范围与解释边界

本项目的基准测试主要研究服务系统行为。EnvBench 在这里提供 trajectory 结构、token 长度与工具等待时间等工作负载信息；正式请求使用确定性 token ID 前缀序列，因此结果不直接对应 EnvBench 的语义任务准确率。

工具等待时间在回放中使用 `duration_scale=0.01`，并受到 `max_sleep_seconds=2` 的限制，因此相关 TTL 与 JCT 指标反映的是当前运行时回放条件。绝对性能同时依赖 GPU、模型、软件栈、服务参数和工作负载。

正式 F 实验没有发生实际 KV 外部迁移，因此本次结果验证了成本感知选择对无效迁移的抑制能力，但没有建立 Frontier-Tail / SSD 外部存储相对 C 的独立加速结论。

## 可复现资源

| 类型 | 路径 |
|---|---|
| 正式工作负载 | `benchmark/workloads/envbench_full_runtime_eligible_max4000.csv` |
| 工作负载审计 | `benchmark/workloads/envbench_full_runtime_eligible_max4000.audit.json` |
| 9 分片清单 | `benchmark/workloads/full_shards_9/manifest.tsv` |
| 最终实验清单 | `benchmark/results/full_envbench_final_20260823/manifest.tsv` |
| 聚合 JSON | `benchmark/results/full_envbench_final_20260823/aggregate_summary.json` |
| 聚合 TSV | `benchmark/results/full_envbench_final_20260823/aggregate_summary.tsv` |
| 聚合 Markdown | `benchmark/results/full_envbench_final_20260823/aggregate_comparison.md` |
| 工作负载构建脚本 | `benchmark/scripts/build_full_envbench_workload.py` |
| 分片构建脚本 | `benchmark/scripts/build_full_envbench_shards.py` |
| 主运行脚本 | `benchmark/scripts/run_full_envbench_vucf.sh` |
| 聚合脚本 | `benchmark/scripts/aggregate_full_envbench.py` |
| Open-loop saturation runner | `benchmark/scripts/run_envbench_runtime_saturation.py` |
| Open-loop saturation 结果 | `benchmark/results/saturation_openloop_20260825/` |
| Open-loop saturation 说明 | `docs/SATURATION_OPEN_LOOP_20260825.md` |

完整冻结交付包 另外保留请求级聚合输出、运行日志与历史证据；GitHub 精简仓库以生成脚本、工作负载、聚合结果和 `manifests/` 为主要复现入口。
