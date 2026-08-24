# 源码来源与版本追溯

本文档记录项目最终源码的仓库关系、Git 提交/标签、关键历史版本节点，以及纯源码、运行时产物与正式基准测试之间的追溯关系。

本项目不是由某个公开 vLLM、Continuum 和 UCM 版本原样拼接得到。最终系统经历了多轮兼容适配、功能扩展和性能优化，因此源码版本以项目冻结的 Git 提交与标签为准；Python 包在运行时打印的版本字符串只作为辅助信息。

## 最终源码版本身份

最终系统包含两个修改后的源码仓库，以及一个负责部署、基准测试和文档的主实验仓库。

| 仓库 | 角色 | 最终分支 | 最终提交 | 最终标签 |
|---|---|---|---|---|
| `YEYVHAIOU/vllm-continuum` | vLLM + Continuum + Dynamic TTL / KVConnector 集成 | `joint-offload-v1` | `6e7d571b831e6e4b82f1b2f8228cc84e9a0261a2` | `vllm-continuum-final-20260820` |
| `YEYVHAIOU/unified-cache-management-continuum` | UCM 兼容适配 + WHEN/WHAT/WHERE 扩展 | `joint-offload-v1` | `be59181f50e515496a7f19a38178c8c2a05cf251` | `ucm-continuum-final-20260822` |
| `YEYVHAIOU/vllm-prefix-experiment` | 部署、基准测试、工作负载、结果与文档 | `main` | 由主仓库 Git 历史定义 | `research-final-20260823` |

两个源码仓库在最终冻结时工作区均为干净状态。部署与正式基准测试默认使用上述两个最终源码版本。

## 三个 GitHub 仓库的关系

GitHub 发布采用三个并列仓库，而不是把两个大型上游源码树再次复制进主实验仓库：

```text
workspace/
├── vllm-prefix-experiment/
├── vllm-continuum/
└── unified-cache-management-continuum/
```

这种划分保留了 vLLM/Continuum 与 UCM 各自的 Git 历史和阶段标签，同时让主仓库保持为部署与实验入口。主仓库脚本默认查找同级的两个源码仓库，也支持通过 `VLLM_REPO` 与 `UCM_REPO` 显式覆盖路径。

## vLLM / Continuum 来源与演进

最终 vLLM 侧源码来自项目中的 `vllm_continuum_repro` 主线。其演进大致为：

```text
vLLM 基线
    ↓
环境与接口适配
    ↓
Continuum 调度
    ↓
固定 TTL
    ↓
Dynamic TTL
    ↓
Prefill / Reload 成本
    ↓
运行时 KV 压力 / 时序提示
    ↓
UCM KVConnector 生命周期集成
    ↓
active load/save + 空传输旁路
    ↓
CUDA Graph / 非 eager 路径恢复
    ↓
最终 vLLM + Continuum 源码
```

Continuum 基础逻辑中存在固定 TTL 阈值。本项目在此基础上加入 `dynamic_ttl_estimator.py`、`TTLEstimatorConfig`、历史阈值、Prefill 曲线和优化后的经验 CDF 等机制。因此最终 vLLM 仓库既不是 原始 vLLM 基线，也不是未经修改的 Continuum 源码快照。

项目还保留一个历史基线仓库：

```text
branch: autodl-baseline-20260803
commit: e1410e026a71abcd04df5322b15aaac0cedded71
```

该仓库用于早期 固定 TTL / Dynamic TTL 对照和 AutoDL 迁移验证，并通过独立 Git bundle 归档。最终 V/U/C/F 基准测试不使用该历史仓库作为 V 的独立代码基座，而是在同一最终 vLLM/Continuum 源码树上通过功能开关形成四种配置，以减少源码版本差异造成的混杂变量。

## UCM 来源与演进

最终 UCM 仓库基于 Unified Cache Management 上游代码继续演进，并针对当前 vLLM V1 生命周期与项目需求加入 KVConnector、存储层和策略修改。

重要阶段如下：

| Git 标签 | 作用 |
|---|---|
| `ucm-dram-uniproc-working-20260818` | DRAM 与调度器/worker 兼容 |
| `joint-offload-pressure-working-20260819` | GPU KV 压力感知 WHEN |
| `joint-offload-temporal-working-20260819` | Continuum 时序提示接入 |
| `joint-offload-transfer-profile-working-20260819` | 数据传输统计 |
| `joint-offload-cost-aware-working-20260819` | 成本感知决策 |
| `joint-offload-cost-full-working-20260819` | 完整成本模型 |
| `tiered-store-working-20260819` | DRAM + SSD TieredStore |
| `sparse-backing-frontier-tail-working-20260819` | Frontier-Tail WHAT |
| `joint-what-cost-aware-working-20260819` | WHAT 保留量与成本模型对齐 |
| `ucm-continuum-final-20260820` | 空传输优化版本节点 |
| `ucm-continuum-final-20260822` | 最终空加载生命周期修正 |

### 历史版本节点 `138769e7...`

```text
138769e7a13fc11b690bc6b954a31764785fb7c5
optimize UCM vLLM connector empty-transfer path
```

该提交保留了空传输优化，是项目历史中的重要节点，对应较早的 `ucm-continuum-final-20260820` 标签。它不是正式 Full EnvBench 实验使用的最终 UCM 源码版本。

### 最终 UCM 版本 `be59181...`

最终源码为：

```text
branch = joint-offload-v1
commit = be59181f50e515496a7f19a38178c8c2a05cf251
tag    = ucm-continuum-final-20260822

subject = fix: initialize KV caches before empty-load fast path
```

该提交在 `138769e7...` 空传输优化的基础上进一步修正空加载初始化顺序，并通过 UCM C76 的 1301/1301 请求回归后用于正式 EnvBench 派生 V/U/C/F 实验。

## 正式基准测试的代码基座

最终四种配置共享同一 vLLM/Continuum 源码版本，并通过调度器与 UCM 开关构造：

```text
V = 最终 vLLM 源码 + FCFS      + UCM 关闭
U = 最终 vLLM 源码 + FCFS      + UCM 开启
C = 最终 vLLM 源码 + Continuum + UCM 关闭
F = 最终 vLLM 源码 + Continuum + UCM 开启
```

U 与 F 使用相同的最终 UCM 源码版本。这样可以避免把历史基线仓库、不同运行时产物、不同 eager 模式或不同 vLLM Git 提交混入最终横向比较。

正式实验使用：

```text
1274 trajectories
13419 tool events
14693 requests per case
9 deterministic shards
concurrency = 128
V/U/C/F
36/36 case PASS
58772 requests total
```

正式结果和聚合口径以 `docs/BENCHMARK.md` 与 `benchmark/results/full_envbench_final_20260823/` 为准。

## 纯源码与运行时产物

完整冻结交付包中的 `source/` 由 `git archive HEAD` 生成，只包含 Git 跟踪文件。这样可以明确回答“最终项目修改了哪些源码”，并避免把工作目录中的构建产物、日志或缓存混入源码快照。

冻结源码文件数量为：

| 源码树 | 文件数 |
|---|---:|
| vLLM + Continuum | 3233 |
| UCM | 438 |
| 合计 | 3671 |

纯源码明确排除：

```text
*.so
*.pyc
__pycache__/
vllm/_version.py
```

这些属于构建或运行时产物，而不是正式 Git 源码。

### 已验证运行快照（Golden Runtime）

vLLM 和 UCM 都依赖原生扩展。完整冻结交付包因此额外保存 `deployment/runtime/golden/`：

```text
已验证运行快照（Golden Runtime）
=
纯 Git 源码
+
已验证的构建与原生运行时产物
```

vLLM 额外运行时文件包括 `_C.abi3.so`、FlashAttention 原生扩展、`cumem_allocator.abi3.so` 与 `_version.py` 等；UCM 额外包含 NFSStore / PCStore 原生扩展。

已验证运行快照（Golden Runtime）与纯源码的一致性检查结果为：

```text
source missing = 0
source changed = 0
runtime extras = 16
```

因此，已验证运行快照（Golden Runtime）保存的是“在原纯源码上增加当前环境所需二进制与运行时产物”的验证快照，而不是另一套修改后的源码。

GitHub 精简仓库不提交已验证运行快照（Golden Runtime）。新机器部署应根据 `docs/DEPLOYMENT.md` 从最终源码版本重新构建与验证；完整冻结交付包用于保留原 AutoDL 软件栈下的精确运行证据。

## 运行时版本与 Git 版本

当前运行环境中：

```text
vllm.__version__ = 0.1.dev10+g05f00f8a8
```

该字符串来自构建或安装阶段生成的 `vllm/_version.py`。由于 `_version.py` 不属于最终纯 Git 源码，它可能反映构建时的版本信息，而不等价于最终项目 HEAD。

项目的版本优先级为：

```text
Git 提交 SHA
    ↓
Git 标签
    ↓
Git 分支
    ↓
运行时 Python 包版本（辅助）
```

因此正式 vLLM 源码版本始终记录为：

```text
6e7d571b831e6e4b82f1b2f8228cc84e9a0261a2
vllm-continuum-final-20260820
```

而 `0.1.dev10+g05f00f8a8` 仅用于记录已验证 Python 运行环境中的包版本。

## 冻结交付、GitHub 与历史归档

项目最终交付同时存在三类用途不同的资产。

| 类型 | 主要用途 | 内容 |
|---|---|---|
| GitHub 仓库 | 代码审查与复现入口 | 源码、部署脚本、基准测试、较小工作负载、聚合结果、文档和清单 |
| 冻结交付包 | 完整研究交付 | 纯源码、已验证运行快照（Golden Runtime）、运行证据、完整清单与归档材料 |
| Git bundle / 阶段快照 | 历史恢复 | 源码历史、早期基线与阶段评审快照 |

GitHub 精简版有意排除模型权重、虚拟环境、原始 EnvBench、SSD 外部存储、已验证运行快照（Golden Runtime）的二进制快照、大型请求级聚合结果和历史原始日志。历史 AutoDL 路径仍可能出现在版本追溯文档、清单或冻结证据中，因为它们记录正式实验实际运行时的位置，而不是 GitHub 检出目录的强制目录结构。

项目曾创建 vLLM、UCM 与基线 Git bundle 归档，并验证基线 bundle 可以恢复到历史实验状态。阶段性 `project_review_snapshot_20260820` 也作为评审与最终冻结证据保留，但这些历史包不替代最终源码仓库。

## 版本与运行证据的对应关系

项目中需要区分三类证据。

**Git 源码**定义代码本身。修改内容、Git 提交/标签和差异均以两个源码仓库为准。

**运行时快照**定义当时已验证机器实际可加载的构建产物和原生扩展。完整冻结交付包中由已验证运行快照（Golden Runtime）保存。

**运行证据**定义服务实际启动时采用的配置和环境。冒烟测试保存实际生效环境、服务日志、KV 传输配置、HTTP 响应与指标等证据。

三者组合形成完整追溯链：

```text
上游与历史源码
        ↓
项目兼容适配与系统集成
        ↓
冻结 Git 提交 / 标签
        ↓
git archive 纯源码
        ↓
已验证运行时产物
        ↓
自动冒烟测试
        ↓
正式基准测试
```

## 完整性清单

主实验仓库通过 `manifests/` 记录源码版本、环境快照与关键文件 SHA256。公开 GitHub 仓库中与部署和基准测试直接相关的主要清单包括：

```text
manifests/final_release_identity.txt
manifests/source_identity.txt
manifests/vllm_version_identity.txt
manifests/gpu_environment.txt
manifests/cuda_environment.txt
manifests/python_environment.txt
manifests/docs.sha256
manifests/deployment_configs.sha256
manifests/deployment_scripts.sha256
manifests/benchmark_scripts.sha256
manifests/benchmark_workloads.sha256
manifests/full_envbench_final.sha256
```

任何正式源码修改都应产生新的 Git 提交，并重新生成受影响的 SHA256 清单。源码实现差异见 `docs/COMPATIBILITY.md`，部署流程见 `docs/DEPLOYMENT.md`，实验结果见 `docs/BENCHMARK.md`。
