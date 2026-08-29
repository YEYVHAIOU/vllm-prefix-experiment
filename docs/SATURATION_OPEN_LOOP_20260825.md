# Open-loop 请求率饱和实验

本文记录 vLLM、vLLM + Continuum 与完整系统在长上下文多轮 Agent 负载下的 open-loop 请求率饱和实验。实验不使用固定客户端并发数作为负载控制变量，而是逐步提高请求到达率，并观测 TTFT、E2E、服务端排队和完成吞吐的变化，用于判断不同系统在持续到达压力下的延迟拐点与过载区间。

本实验是固定并发 Full benchmark 的补充，两者回答的问题不同。固定并发实验关注给定 concurrency 下的吞吐、Prefix Cache 复用和延迟表现；本实验关注在请求持续到达时，系统从何处开始出现延迟快速增长和持续积压。项目总体 benchmark 设计见 [`BENCHMARK.md`](BENCHMARK.md)，部署环境与源码版本分别以 [`DEPLOYMENT.md`](DEPLOYMENT.md) 和 [`SOURCE_PROVENANCE.md`](SOURCE_PROVENANCE.md) 为准。

## 实验设计

实验使用 Full Runtime-Eligible EnvBench-derived workload：

```text
benchmark/workloads/envbench_full_runtime_eligible_max4000.csv
```

完整 workload 包含 1274 条 trajectory、13419 个 tool event 和 14693 个 request。本轮每个请求率点固定接收 3000 个 request，以保证各系统使用相同样本规模，同时避免完整 workload 尾部可运行 trajectory 数逐渐减少对到达率控制造成干扰。

多轮因果关系在 replay 中保持不变。对任一 trajectory，只有前一轮 request 完成并经过对应 tool delay 后，下一轮 request 才进入 ready 状态。因此，load generator 只从当前 ready request 中进行 admission，不提前释放未来 turn。

实验采用 deterministic open-loop rate scheduler。请求按照目标到达率在绝对时间轴上进行 admission，实际绘图和容量判断使用服务端真正收到的 **Actual Arrival Rate**，而不是配置的 Target Rate。正式 runner 位于：

```text
benchmark/scripts/run_envbench_runtime_saturation.py
```

本轮正式点的共同参数如下。

| 参数 | 值 |
|---|---:|
| Model | `Qwen/Qwen3-0.6B` |
| Max model length | 4096 |
| Max input tokens | 4000 |
| Max output tokens | 8 |
| GPU memory utilization | 0.80 |
| Duration scale | 0.01 |
| Max tool sleep | 2 s |
| Metrics interval | 0.2 s |
| Admissions per point | 3000 |
| Client safety fuse (`max_inflight`) | 512 |
| HTTP max connections | 512 |
| HTTP keepalive connections | 128 |

客户端的 `max_inflight=512` 仅作为资源保护，不用于控制 offered load。若某个实验点出现 `peak_inflight=512` 或 `inflight_fuse_waits>0`，说明客户端 safety fuse 已参与限制，该点不用于精确推断服务端容量。

## 实验组

| 代号 | 配置 |
|---|---|
| V | Vanilla vLLM，FCFS，UCM 关闭 |
| U | vLLM + UCM，FCFS，`eager/full` |
| C | vLLM + Continuum，UCM 关闭 |
| F | Continuum + UCM，`joint/cost_full + frontier_tail (K=4)` |

V、C、F 均完成 40、45、50、55、60、80 req/s 六个正式请求率点，且这些点均未触发客户端 safety fuse。U 在较低请求率下即因 eager/full externalization 产生大量 SSD backing 数据，未形成有效 saturation 曲线。

## 指标与容量判断

本实验主要观察四类指标：

| 类别 | 指标 |
|---|---|
| 到达与吞吐 | Actual Arrival Rate、Steady Completed Rate |
| 延迟 | TTFT P50/P95/P99、E2E P50/P95/P99 |
| 排队 | Server Waiting P95/Max、Peak Inflight、Drain Time |
| 调度与缓存 | Prefix Cache token hit rate、TTL hit rate、Preemption |

这里不把最高测试请求率写成“系统最大能力”。文中的 **latency knee** 指 TTFT/E2E 随实际请求到达率开始明显加速增长的实验区间；**持续过载**则结合 arrival 与 steady completion 的差距、服务端 waiting queue 和 drain time 共同判断。该判断是基于本组离散测试点得到的经验区间，不代表对任意模型、硬件或 workload 的普遍上限。

## Vanilla vLLM

| Target | Actual | Steady | TTFT P50 | TTFT P95 | E2E P50 | E2E P95 | Waiting P95 | Peak | Drain |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 40 | 39.80 | 39.72 | 0.035 | 0.118 | 0.091 | 0.291 | 1.0 | 33 | 0.084 |
| 45 | 44.07 | 43.98 | 0.038 | 0.236 | 0.104 | 0.411 | 6.0 | 47 | 0.093 |
| 50 | 48.17 | 48.07 | 0.048 | 0.546 | 0.126 | 0.720 | 23.0 | 61 | 0.087 |
| 55 | 53.00 | 52.90 | 0.084 | 1.112 | 0.231 | 1.286 | 54.2 | 94 | 0.106 |
| 60 | 57.27 | 55.66 | 0.323 | 1.732 | 0.476 | 1.911 | 93.0 | 129 | 0.987 |
| 80 | 72.86 | 63.46 | 2.810 | 5.938 | 2.988 | 6.112 | 324.6 | 387 | 5.910 |

V 在约 40–45 actual req/s 下仍处于低延迟区间；到约 48 req/s 时 TTFT/E2E P95 开始明显抬升，约 53 req/s 时延迟增长进一步加快。到约 57 req/s，steady completion 已开始落后于实际到达率，同时 waiting queue 和 drain time 明显增加。

因此，在本 workload 和硬件条件下，V 的 latency knee 可观察为约 **48–53 actual req/s**，其中约 **53 req/s** 可作为代表性拐点；约 **57 req/s** 后已经出现持续积压迹象。

## vLLM + Continuum

| Target | Actual | Steady | TTFT P50 | TTFT P95 | E2E P50 | E2E P95 | Waiting P95 | Peak | Drain | TTL Hit |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 40 | 39.85 | 39.77 | 0.036 | 0.143 | 0.094 | 0.317 | 3.0 | 33 | 0.081 | 44.11% |
| 45 | 44.16 | 44.05 | 0.042 | 0.429 | 0.112 | 0.580 | 17.4 | 51 | 0.081 | 43.90% |
| 50 | 48.17 | 48.07 | 0.054 | 0.919 | 0.142 | 1.070 | 44.0 | 64 | 0.083 | 42.48% |
| 55 | 52.71 | 52.45 | 0.087 | 3.416 | 0.258 | 3.623 | 76.0 | 111 | 0.134 | 41.54% |
| 60 | 57.31 | 55.96 | 0.134 | 5.095 | 0.310 | 5.242 | 106.0 | 136 | 1.212 | 40.56% |
| 80 | 72.77 | 63.73 | 1.445 | 30.595 | 1.586 | 30.775 | 359.4 | 390 | 6.644 | 36.44% |

所有正式 C 点均记录到 TTL 匹配和非零 preemption，说明 Continuum 调度路径在实验中实际生效。C 的 steady completion 在 40–55 target 区间仍基本跟随 arrival，60 target 后开始出现持续 backlog，因此其 throughput saturation 区域与 V 处于相近量级。

C 与 V 的主要差异出现在延迟分布，而不是简单的吞吐上限。以约 57 actual req/s 为例：

| 指标 | V | C | C 相对 V |
|---|---:|---:|---:|
| TTFT P50 | 0.323 s | 0.134 s | -58.6% |
| TTFT P95 | 1.732 s | 5.095 s | +194.2% |
| E2E P50 | 0.476 s | 0.310 s | -34.8% |
| E2E P95 | 1.911 s | 5.242 s | +174.3% |

实验观察到，高负载下 C 的中位延迟可以低于 V，但 P95/P99 更早、更明显地上升。该现象与 priority-aware scheduling 对请求等待时间进行重新分配的行为一致：部分请求获得更早的执行机会，而另一部分请求可能等待更久。由于本实验没有单独构造用于证明 starvation 机制的对照组，这里将其作为与调度行为一致的解释，而不是对具体因果机制的独立证明。

从 tail latency 看，C 在约 **44–48 actual req/s** 已出现明显抬升；从 steady throughput 和持续 backlog 看，其整体吞吐饱和位置仍与 V 接近。

## Full System

| Target | Actual | Steady | TTFT P50 | TTFT P95 | E2E P50 | E2E P95 | Waiting P95 | Peak | Drain | TTL Hit |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 40 | 39.85 | 39.77 | 0.038 | 0.215 | 0.103 | 0.396 | 6.1 | 40 | 0.099 | 44.36% |
| 45 | 44.08 | 43.99 | 0.048 | 0.545 | 0.126 | 0.707 | 23.2 | 54 | 0.074 | 43.73% |
| 50 | 48.17 | 48.05 | 0.064 | 1.107 | 0.187 | 1.266 | 52.0 | 74 | 0.121 | 42.51% |
| 55 | 52.72 | 51.94 | 0.099 | 4.131 | 0.276 | 4.333 | 82.0 | 123 | 0.650 | 41.68% |
| 60 | 57.36 | 55.56 | 0.181 | 6.832 | 0.357 | 6.878 | 111.0 | 142 | 1.485 | 40.76% |
| 80 | 73.13 | 62.16 | 1.820 | 32.070 | 1.998 | 32.257 | 429.0 | 467 | 8.385 | 35.95% |

F 的整体趋势与 C 接近。40–50 target 时 steady completion 基本能够跟随 arrival；55 target 后排队开始明显增长；60 和 80 target 已进入持续积压区间。与 C 类似，F 在高负载下仍可获得低于 V 的中位延迟，但 P95/P99 tail latency 更早恶化。

本轮所有 F saturation point 的 `ssd_store` 均保持在文件系统最小目录量级（4.0 KiB），未观察到实际 SSD-backed KV migration。因此，本轮 F 的行为主要反映 Continuum 调度与 UCM 决策路径本身，而不能作为“SSD offloading 提升性能”的证据。相较 C，F 没有表现出明显的 capacity 增益，tail latency 还略高。

## UCM eager/full 的资源限制

U 使用 FCFS + eager/full UCM。在 target 15 req/s 的 saturation 尝试中，实验数据盘被 SSD backing 数据耗尽：失败 run 的 `ssd_store` 约为 77 GB，100 GB 数据盘达到满载，实验在生成有效 saturation point 前终止。

因此，本轮没有给出 U 的请求率饱和曲线。该结果只说明：

> 在当前 100 GB 数据盘条件下，eager/full externalization 无法持续承载本轮 open-loop workload 的 saturation 测试。

它不能用于推断 U 的服务端 latency knee。固定并发 Full benchmark 中已经获得的 U 数据仍保留在主 benchmark 结果中，用于分析 eager/full externalization 的实际开销。

## V / C / F 对比

TTFT P95：

| Target | V | C | F |
|---:|---:|---:|---:|
| 40 | 0.118 | 0.143 | 0.215 |
| 45 | 0.236 | 0.429 | 0.545 |
| 50 | 0.546 | 0.919 | 1.107 |
| 55 | 1.112 | 3.416 | 4.131 |
| 60 | 1.732 | 5.095 | 6.832 |
| 80 | 5.938 | 30.595 | 32.070 |

E2E P95：

| Target | V | C | F |
|---:|---:|---:|---:|
| 40 | 0.291 | 0.317 | 0.396 |
| 45 | 0.411 | 0.580 | 0.707 |
| 50 | 0.720 | 1.070 | 1.266 |
| 55 | 1.286 | 3.623 | 4.333 |
| 60 | 1.911 | 5.242 | 6.878 |
| 80 | 6.112 | 30.775 | 32.257 |

在本组测试点中，三种系统的 steady throughput 饱和区间没有出现数量级差异。最明显的差别是 C/F 的延迟分布：在高负载下，它们的 P50 可以低于 V，而 P95/P99 更早进入快速增长区间。因而对 Continuum 和 F，仅用一个“最大请求率”不足以描述系统行为；median latency、tail-latency SLO 与 throughput saturation 需要分开观察。

本实验得到的容量区间可概括为：

| 系统 | Tail-latency knee | 持续 backlog | 说明 |
|---|---|---|---|
| V | 约 48–53 actual req/s | 约 57 actual req/s 后明显 | tail 增长相对平缓 |
| C | 约 44–48 actual req/s | 约 57 actual req/s 后明显 | P50 可优于 V，但 P95/P99 更早恶化 |
| F | 约 44–48 actual req/s | 约 57 actual req/s 后明显 | 趋势接近 C，本轮未触发实际 SSD migration |
| U | 未获得 | 未获得 | eager/full 在本机磁盘条件下先发生资源耗尽 |

这些区间只适用于本实验所使用的模型、GPU、workload 与 replay 配置，不应解释为对应系统的普遍最大容量。

## 无效高压点

V@100、C@100 和 C@120 均触发 `max_inflight=512` 的客户端 safety fuse，因此没有进入正式容量曲线：

| System | Target | Actual | Peak Inflight | Fuse Waits |
|---|---:|---:|---:|---:|
| V | 100 | 78.97 | 512 | 205 |
| C | 100 | 79.59 | 512 | 182 |
| C | 120 | 78.22 | 512 | 541 |

这些点可以说明系统在更高 offered load 下已经存在严重 backlog，但不能用于精确判断服务端 capacity，因为客户端资源保护已经参与限制。

## 与固定并发 benchmark 的关系

固定并发 Full benchmark 与本实验使用相同项目和 workload 系列，但负载生成方式不同：

| 实验 | 负载控制 | 主要回答的问题 |
|---|---|---|
| Full benchmark | 固定 concurrency，trajectory-worker replay | 给定并发下的吞吐、延迟、Prefix Cache 与 UCM 行为 |
| Open-loop saturation | 全局 ready queue + request-rate control | 请求持续到达时的 latency knee、backlog 与吞吐饱和 |

因此，两组实验结果不能直接用单一 throughput 数值互相替代。固定并发实验中 Continuum 的吞吐和 Prefix Cache reuse 优势，与本实验中高负载下出现的 median/tail latency 分化可以同时成立。

## 复现与结果文件

正式 runner：

```text
benchmark/scripts/run_envbench_runtime_saturation.py
```

本轮聚合结果：

```text
benchmark/results/saturation_openloop_20260825/
├── saturation_vcf_final_summary.csv
├── saturation_vcf_final_analysis.xlsx
├── saturation_key_table.png
├── saturation_ttft_p50.png
├── saturation_ttft_p95.png
├── saturation_e2e_p50.png
├── saturation_e2e_p95.png
├── saturation_throughput.png
└── saturation_waiting_p95.png
```

原始 `runtime/runs/`、模型权重、虚拟环境和 SSD backing 数据不属于公开仓库内容。公开仓库保留可复现的 runner、聚合结果和本实验说明；完整部署方式、软件版本和源码来源分别见 [`DEPLOYMENT.md`](DEPLOYMENT.md) 与 [`SOURCE_PROVENANCE.md`](SOURCE_PROVENANCE.md)。
