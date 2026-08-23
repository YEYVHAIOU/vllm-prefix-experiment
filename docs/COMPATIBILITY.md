# 兼容性与源码修改说明（COMPATIBILITY）

## 1. 文档目的

本文档说明本项目最终系统相对于原始开源 vLLM、Continuum 研究代码与 Unified Cache Management（UCM）所进行的修改，并明确区分：

1. **兼容性修改**：为了使原始组件在本项目实际使用的软件栈、vLLM V1 接口和运行环境中能够正确工作而进行的修改；
2. **系统功能扩展**：本课题为实现 Agent 长上下文多轮场景下 KV Cache 优化而新增的机制；
3. **性能工程优化**：在功能正确之后，为降低额外开销、恢复 CUDA Graph、减少无效 KV transfer 等进行的优化；
4. **仅调查但未保留的方案**：曾经分析或实验，但最终没有进入正式系统默认路径的修改。

本文档的目标不是宣称当前代码等价于某个官方发行版，而是给出一条可追溯的工程演进链路。

---

## 2. 最终源码身份

最终系统使用两个主要源码仓库。

### 2.1 vLLM / Continuum

```text
repo:   vllm_continuum_repro
branch: joint-offload-v1
commit: 6e7d571b831e6e4b82f1b2f8228cc84e9a0261a2
tag:    vllm-continuum-final-20260820
```

最终 clean source 交付目录：

```text
source/vllm-continuum/
```

### 2.2 UCM

```text
repo:   unified-cache-management
branch: joint-offload-v1
commit: be59181f50e515496a7f19a38178c8c2a05cf251
tag:    ucm-continuum-final-20260822
```

最终 clean source 交付目录：

```text
source/unified-cache-management/
```

### 2.3 Runtime-reported vLLM 版本

当前已验证运行环境中：

```text
vLLM runtime-reported version:
0.1.dev10+g05f00f8a8
```

这一字符串来自安装 / 构建阶段生成的：

```text
vllm/_version.py
```

因此：

> **最终源码身份以 Git commit / tag 为准；runtime-reported version 仅用于说明当前 Python 安装/构建身份。**

不要用 `0.1.dev10+g05f00f8a8` 替代最终 Git commit 作为源码唯一版本号。

---

## 3. 为什么不能把最终系统描述为“原版开源组件直接组合”

最终系统不是：

```text
原版 vLLM
+
原版 Continuum
+
原版 UCM
```

的简单叠加。

实际演进更接近：

```text
vLLM 基线
    ↓
为当前环境修正 / 适配 vLLM
    ↓
融合 Continuum
    ↓
将固定 TTL 扩展为 Dynamic TTL
    ↓
接入 UCM
    ↓
补齐 DRAMStore / KVConnector 兼容路径
    ↓
加入 runtime context
    ↓
加入 cost-aware / cost_full WHEN
    ↓
加入 TieredStore WHERE
    ↓
加入 Frontier-Tail WHAT
    ↓
优化 empty-transfer path
    ↓
最终 Full System
```

因此，交付时必须同时保留：

- 最终源码；
- 源码身份；
- 兼容性说明；
- 部署运行环境；
- Golden Runtime；
- benchmark 口径。

---

# 第一部分：vLLM / Continuum 侧修改

## 4. Continuum 基础行为与原始固定 TTL

早期 Continuum 代码的核心行为是固定阈值式 Tool Call Residency。

历史代码中可见：

```text
FIXED_THRESHOLD_CONTINUUM = 2.0
```

原始逻辑大致为：

```text
工具执行时间 <= 2 s
    → 保留 / pin

工具执行时间 > 2 s
    → 不继续固定保护
```

历史实现中工具调用时间主要基于：

```text
func_call_to_exec_time
record_func_call_to_exec_time
```

并以历史平均值更新。

这一版本能够体现 Continuum 的核心思想：

> 根据工具执行间隔判断 KV 是否仍值得保留在 GPU。

但它存在两个局限：

1. 固定 2 秒阈值不能适配不同上下文规模；
2. 只使用历史工具时间，没有把 GPU KV pressure、prefill/reload cost、当前 request 生命周期等信息系统化纳入决策。

---

## 5. Dynamic TTL 扩展

本项目在 Continuum 基础上新增 / 完善 Dynamic TTL。

最终源码中包含：

```text
vllm/v1/core/dynamic_ttl_estimator.py
vllm/v1/core/estimate_with_func.py
```

Dynamic TTL 不再只回答：

```text
这个工具历史平均是否小于 2 秒？
```

而是进一步考虑：

```text
工具时间历史
+
当前上下文规模
+
prefill / reload cost
+
经验分布
+
cold start / history threshold
+
运行时压力信息
```

最终配置包括：

```text
CONTINUUM_TTL_CDF_IMPL=optimized
CONTINUUM_HISTORY_THRESHOLD=5
CONTINUUM_DEFAULT_TTL_SECONDS=2.0
CONTINUUM_PREFILL_PROFILE_SCALE=1.0
CONTINUUM_TTL_TIMING_INTERVAL=0
```

这属于：

> **研究功能扩展，而不是为了兼容 vLLM 接口所做的补丁。**

---

## 6. Dynamic TTL CDF 性能优化

Dynamic TTL 的经验 CDF 在早期实现中会产生较大的 Python 端计算开销。

最终版本使用：

```text
optimized empirical CDF
```

并对历史时间序列计算路径进行了优化。

目标是把重复扫描式逻辑从近似：

```text
O(n^2)
```

降低到：

```text
排序 / 单调扫描为主的 O(n log n)
```

并尽量复用 monotonic pointer / 已排序历史信息。

这一类修改属于：

> **性能工程优化。**

它不改变 Dynamic TTL 的语义目标，只降低调度路径自身的 CPU 开销。

---

## 7. Prefill / Reload Cost 建模

为了让 TTL 与 offload decision 不只依赖时间，本项目引入上下文重算成本信息。

当前 prefill profile：

```text
0    -> 0
512  -> 0.016206744
1024 -> 0.038612129
2048 -> 0.095269512
3072 -> 0.164592375
4000 -> 0.234015222
```

最终：

```text
CONTINUUM_PREFILL_PROFILE_SCALE=1.0
```

注意：

> 该基础 profile 历史上来自 RTX 4060 测量。

迁移到 RTX 4090 后做过：

- scale counterfactual；
- 2× / 4× / 8× 敏感性分析；
- 实际 2× 曲线验证。

观察到：

- TTL 会随曲线放大而变长；
- 抢占次数会下降；
- 但整体吞吐没有随之恢复。

因此最终判断：

> prefill profile 会影响 Dynamic TTL，但不是此前性能问题的主要来源。

最终仍采用 `scale=1.0`。

这一点属于当前系统的**已知复现实验限制**：

> 尚未单独在 RTX 4090 上重新完整测量并拟合一条新的 prefill curve。

---

## 8. vLLM Runtime Context 向 UCM 传递

为了让 UCM 的 external backing decision 能利用 Continuum 的时序信息，本项目扩展了 vLLM → UCM runtime context。

最终 UCM Connector 能接收：

```text
kv_pressure
continuum_hints
context_tokens
is_terminal
finish_probability
expected_tool_duration
prefill_reload_cost
```

其中 `set_runtime_context(...)` 会接收当前调度轮次中的 GPU KV pressure 与 Continuum hints。

这一修改的意义是把两个原本独立的问题连接起来：

```text
Continuum:
这段 KV 未来什么时候可能再次使用？

UCM:
现在把它复制到外部存储值不值得？
```

最终形成：

```text
temporal value
+
memory pressure
+
transfer cost
+
recompute cost
```

的联合决策。

这属于：

> **本课题的系统级集成扩展。**

---

# 第二部分：vLLM KVConnector 兼容性修改

## 9. 为什么需要修改 vLLM KVConnector 生命周期

当前项目使用的是 vLLM V1 KV transfer / KVConnector 路径。

UCM upstream 并不是针对本项目最终使用的 vLLM 源码身份原生开发，因此存在：

- connector metadata 生命周期差异；
- worker / scheduler 状态差异；
- CUDA Graph warmup 行为差异；
- 无实际 KV transfer 时仍进入 connector path；
- load / save 调用时机与 UCM connector 内部状态不同步。

这些问题如果不修，会导致：

- 无效 Python 调用；
- 无效 KV load/save；
- warmup 阶段异常；
- connector metadata 尚未 ready 时访问；
- CUDA Graph 被迫关闭或运行不稳定；
- 高并发下额外同步开销。

---

## 10. Active Load / Active Save Protocol

最终 vLLM 代码中加入了对 connector active state 的检查。

例如 model runner 层：

```text
active_load = getattr(kv_connector, "has_active_load", None)

if not callable(active_load) or active_load():
    kv_connector.start_load_kv(...)
```

Attention layer 也加入类似判断：

```text
has_active_load
has_active_save
```

核心作用：

> 当当前调度轮次根本不存在需要 load / save 的 KV 时，不再机械进入完整 connector transfer path。

这一改动跨越：

```text
vLLM model runner
vLLM attention layer
UCM connector active state
```

因此它既是：

- **兼容性修复**
- 也是**性能工程优化**

---

## 11. Empty-transfer Fast Path

最终 UCM Git 历史中存在：

```text
optimize UCM vLLM connector empty-transfer path
```

对应最终 commit：

```text
138769e7a13fc11b690bc6b954a31764785fb7c5
```

优化目标：

```text
没有实际 load/save block
    ↓
跳过无意义的 connector initialization / cache context / synchronization
```

而不是：

```text
每一个 forward 都完整走一遍 transfer protocol
```

最终 fast path 同时在：

- vLLM worker/model-runner 层；
- attention layer；
- UCMConnector；

保持 active state 一致。

这一修改非常重要，因为早期版本即使 UCM 实际 `selected=0`，仍可能引入明显额外延迟。

---

## 12. CUDA Graph / Eager 模式兼容

早期集成阶段为了保证 UCM 路径可运行，曾使用：

```text
--enforce-eager
```

但这会改变 vLLM 正常执行模式，并对性能产生明显影响。

后续通过：

- active transfer bypass；
- warmup 生命周期处理；
- metadata readiness 兼容；
- empty-transfer fast path；

最终恢复：

```text
enforce eager = disabled
```

最终正式部署：

```text
GPU_MEMORY_UTILIZATION=0.80
--scheduling-policy continuum
--enable-prefix-caching
不使用 --enforce-eager
```

因此：

> CUDA Graph 恢复是最终版本的重要工程修复之一。

不能把早期 `enforce-eager=True` 的性能结果与最终正式系统直接混为一谈。

---


## 12.1 2026-08-22 UCM empty-load 生命周期修正

Full EnvBench 正式评测前，UCM Connector 进一步修正了 empty-load fast path 的初始化顺序。

问题表现为：在特定 UCM-only 高并发运行中，load 路径可能在 worker KV cache 尚未从
forward context 完成初始化时提前走 empty-load return；随后 save 侧没有物理 KV cache
可写，但逻辑状态仍可能形成 commit，后续 load 因而出现错误命中并触发缺失 block 的
`KeyError`。

最终修正：

- 在 empty-load fast return 之前先执行 KV cache discovery / initialization；
- 再判断是否可以执行 empty-load fast path；
- 保证逻辑 external-cache 状态与实际物理 KV 数据一致。

该修正已在 UCM C76 回归中验证 1301/1301 requests 成功，并进入最终 UCM 冻结版本：

```text
commit = be59181f50e515496a7f19a38178c8c2a05cf251
tag    = ucm-continuum-final-20260822
```

该节点是最终部署与 Full EnvBench 正式评测使用的 UCM 源码身份。


# 第三部分：UCM 开源版兼容性修改

## 13. UCM upstream 版本白名单问题

当前 UCM patch apply 逻辑显式声明支持：

```text
vLLM 0.9.2
```

而本项目当前 runtime-reported vLLM version：

```text
0.1.dev10+g05f00f8a8
```

因此运行时会出现 warning：

```text
vLLM version ... is not explicitly supported
Supported versions: 0.9.2
```

以及：

```text
Unsupported vLLM version ...
```

但当前项目适配层继续完成 patch，并输出：

```text
All vLLM patches applied successfully for version 0.1.dev10+g05f00f8a8
```

随后已经实机验证：

```text
import
native extension
UCMConnector
UcmTieredStore
API startup
real inference
metrics
stop / port release
```

因此正确表述应为：

> **当前项目使用的是经过本项目兼容性适配并完成实机验证的 UCM + vLLM 联合版本。**

不能写成：

> “UCM 官方原生支持当前 vLLM 版本”。

---

## 14. UCMConnector 与当前 vLLM V1 Request / Block 接口适配

本项目对：

```text
ucm/integration/vllm/ucm_connector.py
```

进行了较大范围适配与扩展。

包括但不限于：

```text
scheduler / worker metadata handling
request block-id mapping
runtime context
load/save lifecycle
active_load
active_save
finished request handling
terminal marker
external backing decision
transfer profiling
```

最终 Connector 不再只是“无条件把 KV 放到外部 store”，而是成为：

```text
vLLM Runtime Context
        ↓
UCM WHEN
        ↓
UCM WHAT
        ↓
UCM WHERE
```

的核心控制层。

---

## 15. Scheduler / Worker 双实例状态兼容

UCM 在当前 vLLM uniproc / V1 路径中会出现 scheduler-side 与 worker-side store / connector 实例。

如果两侧状态完全独立，会导致：

- scheduler 查到的 block 状态与 worker 数据不一致；
- block 已写入但 lookup 不可见；
- DRAM cache 元数据不共享；
- commit 后另一实例仍认为 block 不存在。

因此 DRAMStore 中加入了进程级共享状态，例如：

```text
_PROCESS_SHARED_DRAM_CACHE
_PROCESS_SHARED_CACHED_BLOCKS
```

最终：

```text
self.dram_cache = _PROCESS_SHARED_DRAM_CACHE
self.cached_blocks = _PROCESS_SHARED_CACHED_BLOCKS
```

这属于：

> **为了当前 vLLM uniproc scheduler/worker 生命周期进行的兼容性修改。**

---

## 16. DRAMStore 接口补齐

本项目最终 DRAMStore 不只是简单 Python 字典，而是补齐了当前 UCMConnector 所需接口。

包括：

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

关键配置：

```text
max_cache_size
kv_block_size
max_block_num
```

最终默认逻辑可根据：

```text
max_cache_size // kv_block_size
```

确定最大 block 数。

### 16.1 `create`

为新 block 分配可用空间 / 标记容量。

### 16.2 `lookup`

判断 block 是否已经存在于 external store。

### 16.3 `dump`

把 GPU / worker 侧 block 数据写入 DRAM backing。

### 16.4 `load`

从 DRAM backing 恢复 KV。

### 16.5 `wait`

通过 accelerator Event 等待异步 transfer 完成。

最终 `DramTask` 包含：

```text
task_id
start_event
event
transfer_ms
```

并通过：

```text
event.synchronize()
start_event.elapsed_time(event)
```

记录 transfer time。

### 16.6 `commit`

成功时：

```text
cached_blocks.update(block_ids)
```

失败时释放 / 不确认对应 backing。

这一整套接口补齐是：

> **开源 UCM 与当前 vLLM V1 KVConnector 真正跑通的关键兼容工作之一。**

---

## 17. Accelerator Stream / Event 适配

最终 DRAMStore 支持：

```text
torch.cuda
torch.musa
torch.npu
```

并在当前 RTX 4090 上使用：

```text
torch.cuda
```

数据迁移通过 device Stream / Event 管理。

这样可以：

- 避免把全部 transfer 做成纯同步 host copy；
- 记录真实 transfer timing；
- 与 UCMConnector 的 wait / commit 生命周期对齐。

---

# 第四部分：UCM WHEN —— 何时值得 external backing

## 18. 从 Eager Offload 到 Joint Offload

原始最直接 external backing 可以理解为：

```text
请求产生可保存 KV
    ↓
只要 UCM 开启
    ↓
尝试保存
```

这在高压力 workload 下会造成大量：

```text
GPU → CPU / SSD transfer
```

早期实验发现：

> 外部 backing 本身可能比重新 Prefill 更贵。

因此最终系统没有保留无条件 eager 作为 Full System 默认策略。

---

## 19. GPU KV Pressure

最终 UCMConnector 引入：

```text
kv_pressure
```

范围：

```text
0.0 ~ 1.0
```

并配置：

```text
UCM_PRESSURE_THRESHOLD=0.65
UCM_JOINT_HIGH_THRESHOLD=0.90
UCM_JOINT_AFTER_TTL_THRESHOLD=0.50
```

含义不是简单的一个全局阈值，而是用于：

```text
当前 GPU KV 是否真的已经紧张？
```

如果 GPU pressure 很低：

> external backing 的收益往往不足以抵消 dump/load 成本。

---

## 20. Temporal Hints

Continuum 向 UCM 提供：

```text
continuum_hints
```

其中可包含：

```text
expected_tool_duration
finish_probability
context_tokens
prefill_reload_cost
is_terminal
```

用于回答：

```text
这个 request 未来是否可能回来？
多久后回来？
回来后重算有多贵？
当前是不是已经 terminal？
```

---

## 21. Terminal Request 处理

如果：

```text
is_terminal = true
```

说明该 request 已经没有下一轮需要恢复。

此时 external backing 没有价值。

最终 cost_full decision 会直接跳过：

```text
reason = joint_cost_full_terminal_skip
```

这一修改避免：

> 对已经完成整个 trajectory 的最后一轮 KV 做无意义 dump。

---

## 22. `joint + cost_full`

最终 Full System：

```text
UCM_OFFLOAD_POLICY=joint
UCM_JOINT_DECISION=cost_full
```

传输成本模型：

```text
dump_cost_ms = 2.8000 + 1.5005 * n_blocks
load_cost_ms = 0.3510 + 0.3345 * n_blocks
```

决策不再只使用一个概率，而是把：

```text
GPU pressure
+
future reuse probability
+
tool gap
+
terminal state
+
context size
+
prefill/recompute cost
+
dump cost
+
load cost
```

联合起来。

最终目标是比较：

```text
external backing 的预期价值
```

与：

```text
不保存、未来重新 Prefill 的代价
```

这属于：

> **本项目研究方案的核心功能扩展。**

---

## 23. Zero-evict Skip

在低压力状态下：

```text
没有实际 eviction need
```

cost_full 可以直接：

```text
selected=0
reason=joint_cost_full_zero_evict_skip
```

这不是 UCM 未工作，而是：

> 当前没有值得 external backing 的 KV。

Smoke Test 单请求时出现该情况属于正常现象。

---

# 第五部分：UCM WHAT —— 哪些 KV 值得保存

## 24. 为什么不能简单全量保存

如果一个长上下文 request 有大量 KV blocks：

```text
Full backing
→ 所有候选 block D2H
→ 外部写入
```

会带来明显：

```text
D2H blocks
D2H CUDA time
dump latency
DRAM/SSD 写流量
```

正式实验中这曾成为主要瓶颈之一。

---

## 25. Frontier-Tail Partial Retention

最终：

```text
UCM_WHAT_POLICY=frontier_tail
UCM_RETAIN_BLOCKS=4
```

含义：

> 不默认把完整 Prefix KV 全量 external backing，而只保留 HBM frontier 附近、未来实际恢复最有价值的一小段 tail。

因此它是：

```text
Sparse External KV Backing
```

或：

```text
Frontier-Tail Partial Retention
```

### 25.1 不应称为 Sparse Attention

本项目最终 Frontier-Tail：

- 不改变模型 attention 数学定义；
- 不做 ESA / GSA / KVStar 类 token attention sparsification；
- 不改变单次 forward 中 attention 的语义；
- 改变的是“哪些 KV block 跨 HBM → external tier 被保留”。

因此正式文档应写：

> **Sparse External KV Backing**

而不是：

> **UCM Sparse Attention**

---

## 26. Cost-aware Decision 对 Sparse Backing 的修正

当 WHAT 从 full backing 变成 frontier-tail 后：

```text
实际 dump block 数
```

显著下降。

如果 WHEN 的 cost model 仍按“完整 block 数”估计 transfer cost，就会错误高估 sparse backing 成本。

因此后续加入：

```text
make cost-aware offload aware of sparse KV backing
```

最终 Git 历史：

```text
0dd5c8e...
feat: make cost-aware offload aware of sparse KV backing
```

这一步把：

```text
WHEN 的 cost model
```

与：

```text
WHAT 实际 retention volume
```

重新对齐。

属于：

> **系统集成修正。**

---

# 第六部分：UCM WHERE —— KV 放在哪里

## 27. 原始单级 Store 的局限

只有 DRAMStore 时：

```text
HBM
 ↓
DRAM
```

容量有限。

只有 SSD / NFS store 时：

```text
HBM
 ↓
SSD
```

传输成本更高。

任务要求需要：

```text
HBM
 ↓
DRAM
 ↓
SSD
```

分级体系。

---

## 28. UcmTieredStore

本项目新增：

```text
ucm/store/tieredstore/tieredstore_connector.py
```

核心逻辑：

```text
UcmTieredStore
├── UcmDramStore
└── UcmNfsStore
```

初始化时：

```text
dram_max_cache_size
storage_backends
```

分别传入 DRAM 与 SSD/NFS 子 store。

最终部署：

```text
UCM_TIER_DRAM_MIB=128
UCM_TIER_STREAM_NUMBER=2
UCM_TIER_BUFFER_NUMBER=64
UCM_TIER_USE_DIRECT=false
```

---

## 29. DRAM-first / SSD-second Placement

当前 WHERE policy 是：

```text
DRAM 有容量
    → DRAM

DRAM 容量不足
    → SSD
```

它已经真实验证：

```text
DRAM backing
SSD backing
DRAM load
SSD load
```

因此：

> HBM–DRAM–SSD 分级流转机制已经完成。

但必须注意：

> 当前 WHERE 仍然是容量优先 placement，还不是一个完全 cost-aware 的动态 tier policy。

即当前并没有完整实现：

```text
argmin {
    HBM residency cost,
    DRAM transfer cost,
    SSD transfer cost,
    recompute cost
}
```

这属于未来进一步升级空间，而不是当前部署缺陷。

---

# 第七部分：Native Runtime 与环境兼容

## 30. 为什么 clean source 与 Golden Runtime 分开

最终 clean source 使用：

```text
git archive HEAD
```

生成。

因此不包含：

```text
*.so
*.pyc
__pycache__
vllm/_version.py
```

这些属于：

> build-generated / runtime artifacts。

但当前环境运行 vLLM / UCM 又需要这些 native components。

因此额外保留：

```text
deployment/runtime/golden/
```

---

## 31. Golden Runtime vLLM Native Artifacts

包括：

```text
vllm/_C.abi3.so
vllm/_flashmla_C.abi3.so
vllm/_moe_C.abi3.so
vllm/cumem_allocator.abi3.so
vllm/_version.py
vllm/vllm_flash_attn/
```

其中当前 vLLM native `.so` 包括：

```text
_C.abi3.so
_flashmla_C.abi3.so
_moe_C.abi3.so
cumem_allocator.abi3.so
_vllm_fa2_C.abi3.so
_vllm_fa3_C.abi3.so
```

---

## 32. Golden Runtime UCM Native Artifacts

包括：

```text
ucm/store/nfsstore/ucmnfsstore.cpython-312-x86_64-linux-gnu.so
ucm/store/pcstore/ucmpcstore.cpython-312-x86_64-linux-gnu.so
```

这些 native binary 与当前：

```text
Python 3.12
PyTorch 2.8.0+cu129
CUDA
GCC / GLIBCXX
Linux ABI
```

存在绑定关系。

因此不能把 Golden Runtime 理解为跨平台源码发行物。

---

## 33. `GLIBCXX_3.4.30 not found`

当前 AutoDL 环境中曾出现：

```text
GLIBCXX_3.4.30 not found
```

原因：

> Miniconda 自带 `libstdc++.so.6` 版本低于 UCM native extension 所需 ABI。

最终通过：

```text
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

固定使用系统 libstdc++。

同时：

```text
CUDA_HOME=/usr/local/cuda-12.8
```

也固定到最终部署配置。

这一类修改属于：

> **运行环境兼容性处理。**

不是算法改动。

---

# 第八部分：最终验证环境

## 34. 已验证软件栈

```text
GPU:          NVIDIA GeForce RTX 4090
VRAM:         24564 MiB
Driver:       595.71.05
Python:       3.12.3
PyTorch:      2.8.0+cu129
Torch CUDA:   12.9
CUDA_HOME:    /usr/local/cuda-12.8
nvcc:         CUDA 12.8 / V12.8.93
GCC:          11.2
Transformers: 4.56.2
Tokenizers:   0.22.2
HF Hub:       0.36.2
Model:        Qwen3-0.6B
```

AutoDL `nvidia-smi` 顶部显示的 CUDA Version 可能高于 `CUDA_HOME` Toolkit 版本。

这是正常的：

```text
nvidia-smi CUDA Version
```

表示 driver 支持的最高 CUDA capability，不等价于当前编译 Toolkit。

---

## 35. 最终 Smoke Test 验证

最终 clean deployment 已完成：

```text
SMOKE_HEALTH=PASS
SMOKE_EFFECTIVE_ENV=PASS
SMOKE_HTTP=200
SMOKE_RESPONSE=PASS
SMOKE_METRICS=PASS
SMOKE_RUNTIME_CONFIG=PASS
SMOKE_RUNTIME_ERRORS=NONE
SMOKE_STOP=PASS
SMOKE_PORT_RELEASE=PASS
SMOKE_TEST=PASS
```

说明：

```text
clean source
+
Golden Runtime
+
最终配置
+
兼容环境
```

能够完成真实启动、推理与停止闭环。

---

# 第九部分：修改分类总表

## 36. 兼容性修改

| 修改 | 所在侧 | 类型 |
|---|---|---|
| vLLM V1 KVConnector 生命周期适配 | vLLM + UCM | 兼容 |
| scheduler / worker metadata 生命周期适配 | UCM | 兼容 |
| scheduler / worker DRAM state 共享 | UCM | 兼容 |
| DRAMStore API 补齐 | UCM | 兼容 |
| device Stream / Event transfer | UCM | 兼容 |
| metadata warmup readiness 处理 | vLLM + UCM | 兼容 |
| CUDA Graph / non-eager 路径恢复 | vLLM + UCM | 兼容 + 性能 |
| active load/save protocol | vLLM + UCM | 兼容 + 性能 |
| empty-transfer fast path | vLLM + UCM | 性能 + 兼容 |
| system libstdc++ `LD_PRELOAD` | runtime | 环境兼容 |
| Golden native artifacts | deployment | 构建/运行兼容 |

---

## 37. 系统功能扩展

| 修改 | 作用 |
|---|---|
| Dynamic TTL | 工具调用时序感知 HBM residency |
| optimized empirical CDF | Dynamic TTL 实时估计 |
| prefill/reload cost | 估计 recompute 代价 |
| runtime KV pressure | 判断是否真正需要 backing |
| Continuum hints → UCM | 时序信息跨组件传播 |
| joint offload | pressure + temporal 联合 WHEN |
| `cost_full` | cost-aware external backing |
| terminal skip | trajectory 结束后不再保存 |
| Frontier-Tail | selective external KV retention |
| sparse-aware cost model | WHEN 与 WHAT 对齐 |
| UcmTieredStore | DRAM + SSD 分级 WHERE |

---

## 38. 性能工程优化

包括：

```text
Dynamic TTL CDF 优化
empty-transfer fast path
active transfer bypass
CUDA Graph 恢复
减少无意义 connector 调用
transfer timing profiling
```

这些修改不改变系统目标，但决定最终系统是否能在高并发下具有可接受的运行成本。

---

# 第十部分：调查过但最终未作为默认机制保留的方向

## 39. Hash / Empty Lookup 优化排查

项目过程中排查过：

```text
重复 hash
empty lookup
metadata path
connector no-op path
```

最终 profiling 表明：

> 真正影响明显的不是所有候选小优化，而是空 transfer 生命周期与 connector path 的额外执行。

因此没有把所有实验性 hash / lookup 修改保留为最终默认实现。

正式文档可以表述：

> Investigated but not retained because profiling showed negligible end-to-end benefit.

---

## 40. Prefill Curve 放大实验

曾进行：

```text
2×
4×
8×
```

counterfactual。

并实际运行过：

```text
2×
```

结果说明：

- TTL 变化明显；
- preemption 下降；
- 但整体 throughput 问题没有根治。

因此：

> prefill profile 不是最终性能问题的主要根因。

没有把人为放大的 curve 固定到最终正式配置。

---

## 41. Eager Full Backing

Eager Full UCM 仍可作为实验对照组，但不是最终 Full System 默认策略。

原因：

> 全量 external backing 的 D2H / dump 开销过高。

最终 Full System 使用：

```text
joint + cost_full
+
frontier_tail K=4
+
tiered
```

---

## 42. 官方 UCM Sparse Attention

本项目最终没有把：

```text
ESA
GSA
KVStar
KVComp
```

等 UCM 官方 Sparse Attention 算法接入主系统。

因此最终成果不能表述为：

> “实现了 UCM 官方 Sparse Attention”。

正确表述：

> 实现了面向跨层级 KV Cache 迁移的 Sparse External Backing / Frontier-Tail Partial Retention。

---

# 第十一部分：已知限制与移植风险

## 43. UCM upstream whitelist 不匹配

当前组合经过实机验证，但不属于 UCM upstream 明确 whitelist 的原生版本。

升级 vLLM 后应重新检查：

```text
KVConnector API
scheduler metadata
worker metadata
block lifecycle
attention hook
CUDA Graph warmup
load/save lifecycle
```

---

## 44. Golden Runtime 不保证跨 ABI

如果变化：

```text
Python ABI
PyTorch ABI
CUDA Toolkit
GPU architecture
GCC
GLIBCXX
vLLM native source
UCM native source
```

则应：

```text
从 source 重建 native extension
```

而不是直接复制 Golden `.so`。

---

## 45. Dynamic TTL Prefill Profile

最终 4090 环境没有重新拟合完整 prefill curve。

因此跨 GPU 时建议重新执行：

```text
prefill calibration
```

并更新：

```text
CONTINUUM_PREFILL_PROFILE_SCALE
```

或对应 profile 数据。

---

## 46. TieredStore WHERE 仍为容量优先

当前：

```text
DRAM first
SSD second
```

已经满足 HBM–DRAM–SSD 分级机制需求。

但它不是完整 cost-aware WHERE。

未来可以扩展：

```text
predicted reuse time
transfer latency
DRAM occupancy
SSD occupancy
recompute cost
```

共同决定 placement。

---

# 第十二部分：最终系统边界

## 47. 最终 Full System

最终正式系统定义为：

```text
vLLM
    +
Continuum Dynamic TTL
    +
Runtime KV Pressure / Temporal Hints
    +
UCM Joint WHEN
    +
cost_full Decision
    +
Frontier-Tail WHAT (K=4)
    +
UcmTieredStore WHERE
```

可以简写为：

```text
Dynamic TTL
+
cost_full WHEN
+
Frontier-Tail WHAT
+
TieredStore WHERE
```

---

## 48. 最终应该如何描述本项目修改

推荐表述：

> 本项目并非直接使用原版 vLLM、Continuum 与 UCM，而是在目标 vLLM V1 与 AutoDL CUDA 环境上完成了多项兼容适配，并在 Continuum 的工具调用时序感知机制基础上扩展 Dynamic TTL，将 GPU KV pressure、上下文规模、终止状态、未来复用概率和 prefill/reload cost 传递给 UCM；进一步实现 cost-aware joint offload、Frontier-Tail selective external KV retention，以及 DRAM-first / SSD-second TieredStore，从而形成面向长上下文多轮 Agent 推理的联合 KV Cache 管理系统。

如果强调开源兼容：

> 针对开源版本，主要完成了 vLLM V1 KVConnector 生命周期适配、scheduler/worker metadata 与 DRAM state 兼容、DRAMStore API 补齐、CUDA Stream/Event 数据迁移、warmup / CUDA Graph 兼容，以及空 KV transfer fast path。

如果强调自己的研究扩展：

> 在兼容层之上，进一步实现 Dynamic TTL、temporal hint propagation、cost_full WHEN、Frontier-Tail WHAT 与 TieredStore WHERE。

---

# 第十三部分：结论

最终代码演进可以归纳为三层：

```text
第一层：Compatibility
vLLM V1 ↔ UCM 能正确运行

第二层：System Integration
Continuum temporal value ↔ UCM external backing

第三层：Optimization
WHEN / WHAT / WHERE + fast path
```

因此最终成果的技术边界应清晰写成：

```text
开源基础
    ↓
兼容性适配
    ↓
Dynamic TTL
    ↓
Temporal / Pressure Context
    ↓
Joint Cost-aware Offload
    ↓
Frontier-Tail Partial Retention
    ↓
DRAM / SSD TieredStore
    ↓
Empty-transfer / CUDA Graph 性能修正
```

这也是最终部署版源码与原始开源版本之间最重要的差异。
