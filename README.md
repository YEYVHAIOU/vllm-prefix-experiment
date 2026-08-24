# vLLM + Continuum + UCM

面向长上下文、多轮 Agent 服务的 KV Cache 管理实验系统。

本项目在 vLLM 的 Prefix Cache / PagedAttention 基础上集成 Continuum 风格的时序感知 KV 驻留机制，并接入 Unified Cache Management（UCM）的外部 KV 管理路径，用于研究高并发多轮 Agent 场景中“工具调用间隔导致可复用前缀被提前驱逐”的问题。

最终系统由 **Dynamic TTL + 成本感知 UCM + Frontier-Tail + TieredStore** 组成。项目同时提供源码、部署脚本、EnvBench 派生服务系统回放、压力测试和最终实验结果。

> 本仓库是研究集成项目，不是 vLLM、Continuum、UCM 或 EnvBench 的官方发行版。

## 问题背景

多轮 Agent 请求通常具有很长的共享前缀。一次模型调用结束后，Agent 可能进入文件读取、搜索、Shell 命令或其他工具执行；下一轮请求回来时，大部分历史上下文仍然可以复用。

在高并发服务中，不同 Agent 会共同竞争有限的 GPU KV Cache。工具调用造成的等待间隔使同一 Agent 的前后两轮被其他请求隔开，原本可复用的 KV blocks 可能在下一轮返回前被驱逐，从而重新产生 Prefill 开销。

本项目围绕两个互补问题展开：

- **Continuum / Dynamic TTL**：哪些 KV 值得在 GPU 中继续保留，以及应保留多久；
- **UCM**：当 GPU 内继续保留不划算时，是否需要把 KV 放到外部层级，以及迁移哪些 blocks、放到哪里。

## 最终系统

最终配置为：

```text
Dynamic TTL
    +
cost_full WHEN
    +
Frontier-Tail WHAT (K=4)
    +
TieredStore WHERE (DRAM + SSD)
```

逻辑关系：

```text
Agent 请求
    │
    ▼
GPU KV Cache
    │
    ├── Continuum / Dynamic TTL
    │     估计未来复用间隔与 GPU 内驻留价值
    │
    └── UCM
          ├── WHEN：是否值得迁移
          ├── WHAT：迁移哪些 KV blocks
          └── WHERE：DRAM / SSD
```

Frontier-Tail 是本项目实现的外部 KV 部分保留策略，不改变 Attention 计算语义，也不等价于 UCM 上游 Sparse Attention。

## 仓库结构

项目采用三个 Git 仓库组织：

```text
workspace/
├── vllm-prefix-experiment/
├── vllm-continuum/
└── unified-cache-management-continuum/
```

### `vllm-prefix-experiment`

当前主仓库，负责部署配置与脚本、V/U/C/F 基准测试、EnvBench 派生工作负载、最终聚合结果、环境与源码版本清单，以及部署、实验、兼容性和版本追溯文档。

### `vllm-continuum`

修改后的 vLLM / Continuum 源码：

<https://github.com/YEYVHAIOU/vllm-continuum>

```text
branch = joint-offload-v1
commit = 6e7d571b831e6e4b82f1b2f8228cc84e9a0261a2
tag    = vllm-continuum-final-20260820
```

### `unified-cache-management-continuum`

修改后的 UCM 源码：

<https://github.com/YEYVHAIOU/unified-cache-management-continuum>

```text
branch = joint-offload-v1
commit = be59181f50e515496a7f19a38178c8c2a05cf251
tag    = ucm-continuum-final-20260822
```

该版本包含正式基准测试前完成的空加载 KV Cache 初始化修正。

## 四种实验配置

| 配置 | 调度器 | Continuum | UCM |
|---|---|---:|---:|
| **V** | FCFS | 关闭 | 关闭 |
| **U** | FCFS | 关闭 | eager/full |
| **C** | Continuum | 开启 | 关闭 |
| **F** | Continuum | 开启 | 成本感知 |

V、U、C、F 共享同一最终 vLLM/Continuum 代码基座，U/F 使用同一最终 UCM 代码版本，从而减少不同源码版本造成的实验混杂。

## 正式 EnvBench 派生实验

正式评测使用 **EnvBench 派生的服务系统回放**。它保留 trajectory 结构、输入 token 长度、工具调用顺序与工具等待时间，并使用确定性的 token ID 前缀序列构造请求。

该实验研究服务系统性能，不评估 EnvBench 的语义任务正确率。

| 项目 | 数值 |
|---|---:|
| Trajectories | 1274 |
| 工具事件 | 13419 |
| 每种配置请求数 | 14693 |
| 分片 | 9 |
| 并发度 | 128 |
| V/U/C/F 运行数 | 36 |
| 总请求数 | 58772 |
| 正确性验证 | 36 / 36 PASS |

### 主要结果

| 配置 | Req/s | Prefix Cache 命中率 | Mean TTFT | Mean E2E | Mean JCT |
|---|---:|---:|---:|---:|---:|
| **V** | 79.96 | 65.17% | 0.844 s | 0.976 s | 11.467 s |
| **U** | 25.86 | 67.16% | 2.793 s | 3.808 s | 44.142 s |
| **C** | **116.72** | **86.90%** | **0.302 s** | **0.490 s** | **5.873 s** |
| **F** | 109.24 | 86.89% | 0.327 s | 0.529 s | 6.325 s |

C 相比 V，请求吞吐提高约 **45.97%**，Prefix Cache 命中率从 **65.17%** 提升到 **86.90%**。平均 TTFT、E2E 和 JCT 均明显下降。

U 使用 eager/full KV 外迁后，Prefix Cache 命中率只略有提高，但产生大量外部 KV I/O，吞吐和延迟显著恶化。

F 中的成本感知策略在正式工作负载下没有选择实际 KV 外迁，避免了 U 中的大规模外部数据传输，同时保留了大部分 Continuum 收益。F 的正式结果因此反映“过滤收益不足的迁移”，而不是 SSD 外迁本身带来的加速。

完整方法、尾延迟、KV Cache 利用率、TTL 与 UCM 数据见 [`docs/BENCHMARK.md`](docs/BENCHMARK.md)。

## GPU KV Cache 利用率

正式并发 128 实验中：

| 配置 | Mean | P95 | P99 | Max |
|---|---:|---:|---:|---:|
| V | 10.21% | 31.44% | 50.91% | 62.35% |
| U | 25.20% | 54.08% | 66.21% | 77.10% |
| C | 28.05% | 62.29% | 67.49% | 71.14% |
| F | **28.44%** | 61.03% | 65.31% | **71.69%** |

项目还使用 `min3500` 高压力工作负载测试 C76、C96 和 C128。在 C128 的 C/F 运行中，GPU KV Cache 占用中位数约为 96%，同时所有预期请求仍然成功完成。详细结果见 [`docs/min3500_capacity_C76_C96_C128_final_summary.md`](docs/min3500_capacity_C76_C96_C128_final_summary.md)。

## 快速开始

### 克隆三个仓库

```bash
git clone https://github.com/YEYVHAIOU/vllm-prefix-experiment.git
git clone https://github.com/YEYVHAIOU/vllm-continuum.git
git clone https://github.com/YEYVHAIOU/unified-cache-management-continuum.git
```

保持三个目录同级：

```text
workspace/
├── vllm-prefix-experiment/
├── vllm-continuum/
└── unified-cache-management-continuum/
```

切换源码版本：

```bash
cd vllm-continuum
git checkout vllm-continuum-final-20260820

cd ../unified-cache-management-continuum
git checkout ucm-continuum-final-20260822
```

### 配置本机路径

```bash
export VENV_PATH=/path/to/your/vllm-environment
export MODEL_PATH=/path/to/Qwen3-0.6B
```

如有需要，可覆盖：

```bash
export VLLM_REPO=/path/to/vllm-continuum
export UCM_REPO=/path/to/unified-cache-management-continuum
export CUDA_HOME=/usr/local/cuda-12.8
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

### 启动与验收

```bash
cd vllm-prefix-experiment

bash deployment/scripts/start_full.sh
bash deployment/scripts/status.sh
bash deployment/scripts/smoke_test.sh
bash deployment/scripts/stop.sh
```

完整依赖、原生扩展、配置参数和常见问题见 [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)。

## 基准测试

四种配置位于：

```text
benchmark/configs/V.env
benchmark/configs/U.env
benchmark/configs/C.env
benchmark/configs/F.env
```

正式主运行脚本：

```text
benchmark/scripts/run_full_envbench_vucf.sh
```

快速运行顺序见：

```text
benchmark/README_RUN_ORDER.md
```

正式聚合结果位于：

```text
benchmark/results/full_envbench_final_20260823/
```

主要文件：

```text
aggregate_summary.json
aggregate_summary.tsv
aggregate_comparison.md
manifest.tsv
```

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | 安装、部署、启动、Smoke Test 与移植 |
| [`docs/BENCHMARK.md`](docs/BENCHMARK.md) | 工作负载、指标、聚合方法与正式结果 |
| [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md) | 相对上游的兼容适配与系统修改 |
| [`docs/SOURCE_PROVENANCE.md`](docs/SOURCE_PROVENANCE.md) | 源码来源、提交、标签和冻结版本追溯 |
| [`docs/min3500_capacity_C76_C96_C128_final_summary.md`](docs/min3500_capacity_C76_C96_C128_final_summary.md) | 高压力与并发承载专项实验 |

## 已验证环境

```text
GPU              : NVIDIA GeForce RTX 4090
Python           : 3.12.3
PyTorch          : 2.8.0+cu129
CUDA Toolkit     : 12.8
vLLM runtime     : 0.1.dev10+g05f00f8a8
Model            : Qwen/Qwen3-0.6B
max_model_len    : 4096
max input tokens : 4000
max output tokens: 8
GPU memory util  : 0.80
swap             : 1 GiB
```

具体环境快照保存在 `manifests/`。

## 结果解释范围

本项目的 EnvBench 实验是服务系统回放，不是语义任务准确率评测。请求使用确定性 token ID 前缀序列，因此实验重点是缓存复用、调度、吞吐、延迟和外部 KV 行为。

TTL 命中率基于经过时间缩放和等待上限处理后的回放工具时间计算。`min3500` 工作负载通过人为补长输入制造 GPU KV 压力。正式 F 实验没有发生实际 KV 外迁，因此当前结果没有建立 Frontier-Tail / SSD 外部存储相对 Continuum-only 的独立加速结论。

绝对性能依赖硬件、模型、软件栈、服务参数和工作负载。

## 冻结 release

主仓库 Git 标签：

```text
research-final-20260823
```

完整冻结 release 另外提供归档包，用于保存 GitHub 精简仓库没有纳入的 Golden Runtime、运行证据、历史归档和大型实验资产。

GitHub 仓库不提交模型权重、虚拟环境、SSD backing、Golden Runtime 大型二进制快照和原始 EnvBench 数据。

## 上游与许可证

本项目建立在 vLLM、Continuum 和 Unified Cache Management 等开源项目基础上，并包含项目级兼容适配与研究扩展。

公开发布和再分发时应保留上游许可证、版权声明和必要的 attribution，并遵守相关数据集的再分发条款。UCM 上游 README 中的性能声明属于 UCM 上游项目，不代表本项目正式实验结果。

本仓库不是任何上游项目的官方发行版。
