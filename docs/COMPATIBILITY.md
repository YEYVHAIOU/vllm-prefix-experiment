# 兼容性与源码修改说明

本文档说明本项目相对于上游 vLLM、Continuum 研究代码与 Unified Cache Management（UCM）所进行的兼容性适配、系统集成和性能优化。重点放在最终保留于正式系统中的代码路径；历史实验只在能够解释最终实现来源时保留。

最终系统不是三个上游项目的原样拼接。项目先在目标 vLLM V1 与 RTX 4090 / Python 3.12 / PyTorch 2.8 环境中完成 Continuum 运行和 Dynamic TTL 扩展，随后接入 UCM 的 KVConnector 与外部 KV 存储路径，并继续处理调度器与 worker 生命周期、DRAM/TieredStore、成本感知决策以及空传输路径等兼容与性能问题。

## 最终源码版本身份

| 组件 | 分支 | Git 提交 | Git 标签 |
|---|---|---|---|
| vLLM + Continuum | `joint-offload-v1` | `6e7d571b831e6e4b82f1b2f8228cc84e9a0261a2` | `vllm-continuum-final-20260820` |
| UCM | `joint-offload-v1` | `be59181f50e515496a7f19a38178c8c2a05cf251` | `ucm-continuum-final-20260822` |

当前运行环境中的 `vllm.__version__` 为 `0.1.dev10+g05f00f8a8`。该值来自构建或安装阶段生成的 `_version.py`，用于描述 Python 运行包版本；正式源码版本以 Git 提交与标签为准。

## 系统演进

最终代码路径可以概括为：

```text
vLLM 基线
    ↓
Continuum 调度
    ↓
Dynamic TTL
    ↓
运行时 KV 压力 / 时序提示
    ↓
UCM KVConnector 兼容
    ↓
成本感知 WHEN
    ↓
Frontier-Tail WHAT
    ↓
DRAM + SSD TieredStore WHERE
    ↓
空传输 / 空加载 / CUDA Graph 修正
    ↓
最终完整系统
```

这些修改大致分为三层：组件接口与运行时兼容、本项目联合 KV Cache 管理机制、以及用于降低附加开销的性能工程优化。

## vLLM / Continuum 侧修改

### Dynamic TTL

Continuum 基础实现使用固定阈值判断工具调用期间的 KV 驻留，历史代码中存在 `FIXED_THRESHOLD_CONTINUUM = 2.0`。本项目在此基础上扩展 Dynamic TTL，使保留时间能够结合在线历史和上下文重算成本变化，而不是始终使用固定 2 秒阈值。

最终实现主要位于：

```text
vllm/v1/core/dynamic_ttl_estimator.py
vllm/v1/core/estimate_with_func.py
```

正式配置为：

| 参数 | 最终值 |
|---|---:|
| `CONTINUUM_TTL_CDF_IMPL` | `optimized` |
| `CONTINUUM_HISTORY_THRESHOLD` | `5` |
| `CONTINUUM_DEFAULT_TTL_SECONDS` | `2.0` |
| `CONTINUUM_PREFILL_PROFILE_SCALE` | `1.0` |
| `CONTINUUM_TTL_TIMING_INTERVAL` | `0` |

估计器综合工具调用历史、冷启动默认 TTL、上下文规模以及 Prefill / Reload 重算成本等信息估计后续复用间隔。项目还优化了经验 CDF 计算路径，减少调度热路径中的重复扫描开销。

### Prefill / Reload 成本

Dynamic TTL 与后续 UCM 决策都需要估计“KV 丢失后重新 Prefill”的代价。项目冻结的 Prefill 曲线为：

```text
0    -> 0
512  -> 0.016206744
1024 -> 0.038612129
2048 -> 0.095269512
3072 -> 0.164592375
4000 -> 0.234015222
```

这条基础曲线历史上来自 RTX 4060 测量。迁移到 RTX 4090 后，项目进行过 2× / 4× / 8× 反事实放大分析，并实际运行过 2× 配置。放大曲线会延长 TTL 并减少部分抢占，但没有解释此前的主要吞吐差异，最终正式配置保持 `CONTINUUM_PREFILL_PROFILE_SCALE=1.0`。

因此这条曲线是当前可复现配置的一部分，但并非针对 RTX 4090 重新完整拟合的测量结果。跨 GPU 部署时，应重新测量或至少重新校准缩放参数。

### 运行时上下文

为了让 UCM 的迁移决策使用 Continuum 提供的时序价值，本项目扩展了 vLLM → UCM 的运行时上下文。最终 KVConnector 可接收：

```text
kv_pressure
continuum_hints
context_tokens
is_terminal
finish_probability
expected_tool_duration
prefill_reload_cost
```

这些信息把“多久后可能复用”“当前 GPU KV 是否紧张”“重新 Prefill 的代价有多高”和“外部迁移的代价有多高”连接到同一条决策路径中。

### KVConnector 活跃状态与空传输路径

UCM 上游实现与本项目使用的 vLLM V1 生命周期并不完全一致。早期集成中，即使一次前向过程没有实际 KV load/save，也可能进入 KVConnector 初始化、元数据准备和同步路径，从而产生额外的 Python 与 CUDA 开销，并影响 warmup 和 CUDA Graph。

最终 vLLM 与 UCM KVConnector 增加了 active load/save 协议。model runner 与 attention layer 在执行 `start_load_kv` 或保存相关逻辑前，会检查当前 KVConnector 是否真的存在数据传输。UCM 侧同步维护 `has_active_load` / `has_active_save` 等状态，使两侧在空传输场景中保持一致。

由此形成空传输快速路径：当本轮没有实际 load/save block 时，跳过不必要的数据传输生命周期，而不是每次前向过程都执行完整 KVConnector 路径。

### CUDA Graph 兼容

项目早期为了保证 UCM 路径可运行，曾使用 `--enforce-eager`。后续通过活跃传输旁路、元数据就绪处理、warmup 生命周期修正和空传输快速路径，最终恢复非 eager 的 CUDA Graph 执行路径。

正式部署中：

```text
--scheduling-policy continuum
--enable-prefix-caching
gpu_memory_utilization = 0.80
enforce_eager = false
```

因此早期强制 eager 实验与最终正式基准测试不属于同一种执行模式。

## UCM 与 vLLM V1 的兼容性适配

### 上游版本支持范围

当前 UCM 上游补丁逻辑显式面向 vLLM 0.9.2，而本项目运行时报告的 vLLM 版本为 `0.1.dev10+g05f00f8a8`。启动时可能出现版本不在显式支持列表中的警告，但项目适配层能够完成补丁，并已通过导入、原生扩展加载、API 启动、真实推理、指标读取和正式实验验证。

因此当前组合属于“基于 UCM 上游代码、针对本项目 vLLM V1 源码版本完成兼容适配的联合版本”，而不是 UCM 上游对当前 vLLM Git 提交的原生发行支持。

### UCMConnector 生命周期

`ucm/integration/vllm/ucm_connector.py` 是联合系统的核心适配层。项目修改覆盖调度器/worker 元数据、请求与 block 映射、运行时上下文、load/save 生命周期、请求结束处理、terminal 标记、外部 KV 存储决策以及数据传输统计。

最终控制链为：

```text
vLLM 运行状态
        ↓
UCM WHEN
        ↓
UCM WHAT
        ↓
UCM WHERE
```

这使 UCM 不再只是一个无条件外部存储，而是参与每次候选 KV 迁移的选择。

### 调度器与 worker 共享状态

在当前 vLLM V1 单进程路径中，调度器侧与 worker 侧可能分别创建存储或 KVConnector 实例。如果两侧独立维护外部缓存元数据，会出现 lookup、commit 与物理 KV 数据可见性不一致的问题。

项目在 DRAMStore 中引入进程级共享状态：

```text
_PROCESS_SHARED_DRAM_CACHE
_PROCESS_SHARED_CACHED_BLOCKS
```

这样，同一进程内的调度器与 worker 可以共享缓存元数据与 block 外部副本。该修改属于针对当前 vLLM 调度器/worker 生命周期的兼容性处理。

### DRAMStore 接口

为适配当前 UCMConnector，项目补齐了 DRAMStore 的主要数据和生命周期接口：

```text
create
lookup
prefetch
load
dump
fetch_data
dump_data
wait
commit
check
```

容量由 `max_cache_size`、`kv_block_size` 与 `max_block_num` 管理。数据迁移通过加速设备的 Stream/Event 组织，并记录传输时间；在 RTX 4090 环境中实际使用 `torch.cuda`。

`commit` 只在数据传输成功后更新已缓存 block 的元数据，从而保持逻辑外部缓存状态与实际存储数据一致。

### 空加载初始化修正

正式 EnvBench 派生实验前，UCM-only 高并发回归暴露出一个空加载生命周期问题：load 路径可能在 worker KV cache 尚未从前向上下文完成初始化时提前返回，随后 save 侧缺少物理 KV cache，但逻辑状态仍可能形成 commit，后续 load 会遇到缺失 block。

最终修复调整了初始化顺序：先从前向上下文完成 KV Cache 发现与初始化，再判断是否可以进入空加载快速路径。该修复在 C76 UCM 回归中完成 1301/1301 个请求，并进入最终 UCM 提交：

```text
be59181f50e515496a7f19a38178c8c2a05cf251
fix: initialize KV caches before empty-load fast path
```

## UCM 策略扩展

### WHEN：`joint + cost_full`

最直接的 eager 外迁会对所有合格 KV 候选尝试保存。项目压力实验表明，这种方式可能产生远高于重算收益的 D2H / SSD 数据传输开销。因此最终完整系统使用：

```text
UCM_OFFLOAD_POLICY=joint
UCM_JOINT_DECISION=cost_full
```

决策综合 GPU KV 压力、未来复用概率、工具等待间隔、终止状态、上下文规模、Prefill/重算成本以及外部数据传输成本。

当前传输成本模型为：

```text
dump_cost_ms = 2.8000 + 1.5005 * n_blocks
load_cost_ms = 0.3510 + 0.3345 * n_blocks
```

相关压力阈值为：

| 参数 | 值 |
|---|---:|
| `UCM_PRESSURE_THRESHOLD` | `0.65` |
| `UCM_JOINT_HIGH_THRESHOLD` | `0.90` |
| `UCM_JOINT_AFTER_TTL_THRESHOLD` | `0.50` |

终止请求没有后续复用，因此 `is_terminal=true` 时可以直接跳过外部存储。低压力且不存在实际驱逐需求时，`cost_full` 也可能得到 `selected=0`；这是成本判断结果，而不是 KVConnector 未启用。

### WHAT：Frontier-Tail 部分保留

full WHAT 会把全部候选 KV blocks 迁移到外部存储。对于长前缀，这会迅速放大数据传输量。项目新增：

```text
UCM_WHAT_POLICY=frontier_tail
UCM_RETAIN_BLOCKS=4
```

Frontier-Tail 只在 WHEN 选择迁移后保留有限数量的 frontier/tail blocks，以降低外部 KV 存储量。该策略作用于跨存储层级的 KV 保留，不改变单次前向过程的 Attention 语义。

当 WHAT 从 full 变为部分保留后，WHEN 的成本模型也必须使用实际保留量，而不能继续按全部候选 blocks 估计成本。项目因此加入与 Frontier-Tail 对齐的成本计算，使 `cost_full` 使用真实的预期传输规模。

### WHERE：TieredStore

项目新增 `UcmTieredStore`，把外部 KV 存储从单级扩展为：

```text
HBM
 ↓
DRAM
 ↓
SSD
```

当前实现由 `UcmDramStore` 与 `UcmNfsStore` 组合。正式配置为：

| 参数 | 值 |
|---|---:|
| DRAM 容量 | 128 MiB |
| 传输 Stream 数 | 2 |
| 传输 Buffer 数 | 64 |
| `use_direct` | false |

当前放置策略采用 DRAM-first / SSD-second：DRAM 有空间时优先放入 DRAM，空间不足时再进入 SSD。该路径已经验证 create、dump 与 load 可以跨 DRAM / SSD 正常工作。

目前 WHERE 仍是容量优先策略，尚未在 HBM、DRAM、SSD 与重算成本之间做统一的成本优化放置。这属于当前实现边界。

## 运行时与原生扩展兼容

vLLM 与 UCM 都包含原生扩展。完整冻结交付包因此区分纯 Git 源码与已验证运行快照：

```text
已验证运行快照（Golden Runtime）
=
纯 Git 源码
+
validated generated/native artifacts
```

vLLM 运行快照额外包含 `_C.abi3.so`、FlashAttention 扩展、`_version.py` 等构建产物；UCM 运行快照包含 NFSStore / PCStore 原生扩展。这些二进制文件与 Python ABI、PyTorch、CUDA、GCC/GLIBCXX 以及 GPU 架构存在绑定关系。

AutoDL 环境中曾出现：

```text
GLIBCXX_3.4.30 not found
```

原因是 Miniconda 提供的 `libstdc++.so.6` 与 UCM 原生扩展所需 ABI 不匹配。最终验证环境使用：

```text
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
CUDA_HOME=/usr/local/cuda-12.8
```

GitHub 精简仓库不提交已验证运行快照（Golden Runtime）。在新机器上部署时，应从冻结源码重新构建原生组件，并重新执行冒烟测试。

## 修改分类

| 类别 | 主要修改 |
|---|---|
| 兼容性适配 | vLLM V1 KVConnector 生命周期、调度器/worker 元数据、DRAM 共享状态、DRAMStore API、原生 ABI / libstdc++ 处理 |
| 系统集成 | Dynamic TTL、运行时压力与时序提示、`joint + cost_full`、Frontier-Tail、TieredStore |
| 性能优化 | 经验 CDF 优化、active load/save、空传输快速路径、空加载初始化修正、CUDA Graph 恢复、数据传输统计 |

兼容性适配保证上游组件在当前源码与运行环境中能够正确协作；系统集成定义联合 KV Cache 管理机制；性能优化用于避免 KVConnector 与估计器本身成为新的服务瓶颈。

## 当前实现边界

当前组合经过 RTX 4090 / Python 3.12 / PyTorch 2.8 / CUDA 12.8 环境的实机验证。升级 vLLM、UCM 或运行栈后，需要重新检查 KVConnector API、调度器/worker 元数据、block 生命周期、Attention hooks、warmup / CUDA Graph 和原生 ABI。

Dynamic TTL 使用的基础 Prefill 曲线没有在 RTX 4090 上重新完整拟合，跨 GPU 时建议重新校准。TieredStore 当前采用 DRAM-first / SSD-second 的容量优先放置，尚未实现结合预测复用时间、传输延迟、存储层占用与重算成本的统一 WHERE 决策。

项目没有把 UCM 上游的 ESA、GSA、KVStar、KVComp 等 Sparse Attention 算法接入正式系统。Frontier-Tail 是外部 KV 的部分保留策略，属于存储层级管理。

正式 EnvBench 派生实验中，F 的 `cost_full` 没有选择实际 KV 外部迁移。因此该实验验证的是联合成本判断能够避免收益不足的迁移，而不是 Frontier-Tail / SSD 外部存储相对 仅 Continuum 配置 的独立性能增益。相关结果见 `docs/BENCHMARK.md`。

## 最终系统

最终完整系统可以概括为：

```text
vLLM V1
    +
Continuum Dynamic TTL
    +
运行时 KV 压力 / 时序提示
    +
UCM joint cost_full WHEN
    +
Frontier-Tail WHAT (K=4)
    +
DRAM-first / SSD-second TieredStore WHERE
    +
KVConnector 生命周期与空传输兼容修正
```

从工程结构上看，系统由“vLLM/UCM 兼容适配 → 时序与成本联合决策 → 运行路径性能优化”三层组成。源码演进、最终 Git 提交/标签与历史版本节点见 `docs/SOURCE_PROVENANCE.md`。
