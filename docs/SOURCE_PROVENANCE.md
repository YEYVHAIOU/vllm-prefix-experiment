# 源码来源与版本追溯说明（SOURCE_PROVENANCE）

## 1. 文档目的

本文档用于记录本项目最终交付源码的来源、版本身份、仓库演进关系和运行时生成产物边界。

需要特别说明：

> 本项目最终系统并不是直接由“某个公开 vLLM release + 某个公开 Continuum release + 某个公开 UCM release”无修改拼接得到。

实际工程经历了多阶段适配、集成和优化，因此最终交付必须以**本项目冻结的 Git commit / tag**作为正式源码身份，而不能仅依赖 Python 包运行时打印出的版本号。

本文档回答以下问题：

1. 最终系统使用哪两个源码仓库；
2. 正式 commit / tag 是什么；
3. 历史 baseline 与最终主线是什么关系；
4. `source/` 与 `deployment/runtime/golden/` 有什么区别；
5. 为什么 runtime-reported vLLM version 与最终 Git commit 不完全一致；
6. 哪些内容属于源码，哪些属于构建或运行产物；
7. 如果以后上传 GitHub，应以什么作为版本基准。

---

## 2. 最终源码总览

最终系统由两个主要源码仓库组成：

```text
vLLM / Continuum:
    /root/autodl-tmp/vllm_workspace/src/vllm_continuum_repro

UCM:
    /root/autodl-tmp/vllm_workspace/src/unified-cache-management
```

clean deployment 中对应：

```text
vllm_deploy_clean_20260820/
└── source/
    ├── vllm-continuum/
    └── unified-cache-management/
```

最终系统的正式源码身份：

```text
vLLM / Continuum
branch: joint-offload-v1
commit: 6e7d571b831e6e4b82f1b2f8228cc84e9a0261a2
tag:    vllm-continuum-final-20260820

UCM
branch: joint-offload-v1
commit: be59181f50e515496a7f19a38178c8c2a05cf251
tag:    ucm-continuum-final-20260822
```

两个仓库在最终冻结时均为 clean working tree。

---

# 第一部分：最终 vLLM / Continuum 主线

## 3. 正式仓库

路径：

```text
/root/autodl-tmp/vllm_workspace/src/vllm_continuum_repro
```

最终：

```text
branch = joint-offload-v1
HEAD   = 6e7d571b831e6e4b82f1b2f8228cc84e9a0261a2
tag    = vllm-continuum-final-20260820
```

这是最终部署、Golden Runtime 和正式 benchmark 应使用的 vLLM / Continuum 源码。

---

## 4. vLLM / Continuum 的历史演进

本项目最初并不是从最终三模块联合系统开始。

工程演进可概括为：

```text
vLLM 基础环境
    ↓
为当前 Python / Torch / CUDA / GPU 环境进行修正
    ↓
融合 Continuum
    ↓
固定 TTL
    ↓
Dynamic TTL
    ↓
prefill / reload cost
    ↓
runtime KV pressure
    ↓
Continuum temporal hints
    ↓
UCM KVConnector lifecycle integration
    ↓
active load/save
    ↓
empty-transfer bypass
    ↓
CUDA Graph / non-eager 恢复
    ↓
最终 vLLM / Continuum 主线
```

因此：

> 最终 `vllm_continuum_repro` 不是简单等价于公开 vLLM 源码，也不是单纯把 Continuum 的少量文件复制进去。

它是本项目最终联合系统中的 vLLM 侧实现。

---

## 5. Dynamic TTL 的来源关系

历史 Continuum 基础逻辑包含固定 TTL：

```text
FIXED_THRESHOLD_CONTINUUM = 2.0
```

并根据工具历史执行时间决定是否继续 pin。

本项目在该基础上进一步加入：

```text
dynamic_ttl_estimator.py
TTLEstimatorConfig
history threshold
default TTL
prefill profile
optimized empirical CDF
context-sensitive reload cost
```

因此正式源码中的 Dynamic TTL 属于：

> 基于 Continuum 核心思想进一步实现和扩展的项目版本。

---

## 6. 历史 baseline 仓库

历史 baseline 路径：

```text
/root/autodl-tmp/vllm_workspace/src/vllm_continuum_baseline
```

历史身份：

```text
branch:
autodl-baseline-20260803

commit:
e1410e026a71abcd04df5322b15aaac0cedded71
```

该仓库曾用于：

- 早期 Continuum baseline；
- fixed TTL 实验；
- baseline / dynamic 对照；
- 迁移 AutoDL 后的复现实验。

历史工作树还曾存在一个用于：

```text
CONTINUUM_FIXED_THRESHOLD_SECONDS
```

可配置化的本地 patch。

该 baseline 已经通过独立 Git bundle 与 patch 归档，可用于历史实验复现。

但是：

> **它不是最终部署 Full System 的源码。**

最终四模式 benchmark 如果采用统一最终代码基座，应避免重新使用该历史 baseline repo 作为 Vanilla vLLM 基座，以免引入代码版本差异。

---

## 7. 为什么最终 benchmark 应统一代码基座

早期实验中曾存在：

```text
baseline repo
dynamic repo
UCM launcher
Continuum launcher
```

并行演进。

这会带来潜在混杂变量：

```text
不同 vLLM commit
不同 runtime artifacts
不同 scheduler path
不同 eager setting
不同 GPU memory utilization
```

因此最终公平比较应定义为：

```text
V = 最终 vLLM / Continuum 源码 + FCFS + UCM OFF
U = 最终 vLLM / Continuum 源码 + FCFS + UCM ON
C = 最终 vLLM / Continuum 源码 + Continuum + UCM OFF
F = 最终 vLLM / Continuum 源码 + Continuum + UCM ON
```

即：

> 四种模式尽量共享同一最终源码树，只通过明确的功能开关改变模块。

历史 baseline repo 只用于历史结果追溯，不作为最终发布 benchmark 的默认代码基座。

---

# 第二部分：UCM 源码来源

## 8. 最终 UCM 主线

路径：

```text
/root/autodl-tmp/vllm_workspace/src/unified-cache-management
```

最终：

```text
branch = joint-offload-v1
HEAD   = be59181f50e515496a7f19a38178c8c2a05cf251
tag    = ucm-continuum-final-20260822
```

这是最终部署中使用的 UCM 源码。

---

## 9. UCM 最终 Git 演进节点

最终 UCM Git 历史中保留了多个关键阶段 tag。

已确认包括：

```text
ucm-dram-uniproc-working-20260818

joint-offload-pressure-working-20260819
joint-offload-temporal-working-20260819
joint-offload-transfer-profile-working-20260819
joint-offload-cost-aware-working-20260819
joint-offload-cost-full-working-20260819

tiered-store-working-20260819

sparse-backing-frontier-tail-working-20260819

joint-what-cost-aware-working-20260819

ucm-continuum-final-20260820
```

这些 tag 形成了一条系统演进线：

```text
DRAM compatibility
    ↓
pressure-aware WHEN
    ↓
temporal-aware WHEN
    ↓
transfer profiling
    ↓
cost-aware WHEN
    ↓
cost_full
    ↓
TieredStore WHERE
    ↓
Frontier-Tail WHAT
    ↓
WHAT-aware cost model
    ↓
empty-transfer optimization
    ↓
final
```

---

## 10. UCM 最终 commit

最终 HEAD：

```text
138769e7a13fc11b690bc6b954a31764785fb7c5
```

最终 commit message：

```text
optimize UCM vLLM connector empty-transfer path
```

它是在此前：

```text
0dd5c8e...
feat: make cost-aware offload aware of sparse KV backing
```

基础上继续优化。

因此最终 UCM 同时包含：

```text
WHEN
joint / cost_full

WHAT
frontier_tail

WHERE
tiered

performance
empty-transfer fast path
```

---

## 11. UCM upstream 与项目版本的关系

当前 UCM 仓库不能简单描述为：

```text
官方 UCM 某版本原样使用
```

原因包括：

- 当前项目对 DRAMStore、connector、scheduler/worker 生命周期、TieredStore、WHEN/WHAT 等做了扩展；
- UCM patch whitelist 明确支持的是特定 vLLM version；
- 当前项目 vLLM runtime identity 不在该 whitelist 中；
- 最终系统依赖本项目完成的兼容性适配。

因此交付时推荐描述：

> 本项目 UCM 源码基于 Unified Cache Management 开源代码演进，并针对当前 vLLM V1 和项目运行环境进行了兼容性适配，同时加入了本课题所需的 joint offload、cost_full、Frontier-Tail 和 TieredStore 扩展。

详细差异见：

```text
docs/COMPATIBILITY.md
```

---

# 第三部分：Clean Source 生成方式

## 12. `source/` 不是直接复制工作目录

最终 clean source 使用：

```bash
git archive HEAD
```

生成。

例如：

```text
git -C "$VLLM" archive HEAD
git -C "$UCM" archive HEAD
```

然后解包到：

```text
source/vllm-continuum/
source/unified-cache-management/
```

这样可以确保：

- 只包含 Git tracked files；
- 不混入 runtime logs；
- 不混入 `.pyc`；
- 不混入 `__pycache__`；
- 不混入 native `.so`；
- 不混入构建生成 `_version.py`；
- 不混入实验过程临时文件。

---

## 13. Clean Source 文件数量

最终冻结：

```text
vllm source files = 3233
ucm source files  = 438
```

总计：

```text
3671 tracked source files
```

对应 manifest：

```text
manifests/vllm_source.sha256
manifests/ucm_source.sha256
manifests/source_file_counts.txt
```

---

## 14. Clean Source 中明确排除的内容

最终 `source/` 不包含：

```text
*.so
*.pyc
__pycache__/
vllm/_version.py
```

这些不是正式 Git source。

---

# 第四部分：Runtime Generated Artifacts

## 15. 为什么源码不能直接在当前机器零成本运行

vLLM 和 UCM 都包含 native extension。

当前运行需要：

```text
C++ / CUDA extension
FlashAttention extension
UCM native NFS / PC store extension
```

而这些不是 Git tracked source。

因此只复制纯源码到一台机器后：

> 需要重新 build，或者使用与目标 ABI 完全匹配的已构建 runtime artifacts。

---

## 16. Golden Runtime 的作用

为了保留当前已经验证成功的完整运行状态，项目另外建立：

```text
deployment/runtime/golden/
```

结构：

```text
golden/
├── vllm-continuum/
└── unified-cache-management/
```

它的构造原则：

```text
Golden Runtime
=
Clean Git Source
+
当前已验证环境所需 generated/native artifacts
```

---

## 17. Golden Runtime 中 vLLM 额外文件

相对 pure source，vLLM runtime 增加：

```text
vllm/_C.abi3.so
vllm/_flashmla_C.abi3.so
vllm/_moe_C.abi3.so
vllm/cumem_allocator.abi3.so
vllm/_version.py
vllm/vllm_flash_attn/
```

其中 native extension 包括：

```text
_C.abi3.so
_flashmla_C.abi3.so
_moe_C.abi3.so
cumem_allocator.abi3.so
_vllm_fa2_C.abi3.so
_vllm_fa3_C.abi3.so
```

---

## 18. Golden Runtime 中 UCM 额外文件

包括：

```text
ucm/store/nfsstore/ucmnfsstore.cpython-312-x86_64-linux-gnu.so

ucm/store/pcstore/ucmpcstore.cpython-312-x86_64-linux-gnu.so
```

---

## 19. Golden Runtime Source Integrity

Golden Runtime 在生成后进行了 source integrity 比较：

```text
source missing = 0
source changed = 0
runtime extras = 16
```

因此 Golden Runtime 没有修改 clean source。

它只是额外添加：

```text
16 个 generated/runtime artifacts
```

对应：

```text
manifests/golden_runtime_delta_files.txt
```

---

## 20. Golden Runtime Hash

分别有：

```text
manifests/golden_vllm_runtime.sha256
manifests/golden_ucm_runtime.sha256
```

已经验证通过。

因此 Golden Runtime 可以作为：

> 当前 RTX 4090 / Python 3.12 / Torch / CUDA 软件栈下的 exact verified runtime snapshot。

---

# 第五部分：Runtime Version 与 Git Version

## 21. 为什么 `vllm.__version__` 不是最终源码 commit

运行时：

```text
vllm.__version__
=
0.1.dev10+g05f00f8a8
```

但正式最终 Git HEAD：

```text
6e7d571b831e6e4b82f1b2f8228cc84e9a0261a2
```

两者并不相同。

原因：

> vLLM 的 Python version string 来自安装 / 构建阶段生成的 `_version.py`。

`_version.py` 本身在 `.gitignore` 中被排除。

同时 vLLM `setup.py` 中：

```text
get_version(write_to="vllm/_version.py")
```

会在 build/install 阶段生成该文件。

因此当前 `_version.py` 可能保留了某个 build-time Git identity。

---

## 22. 正确版本优先级

正式交付时版本优先级应为：

```text
第一优先级:
Git commit SHA

第二优先级:
Git tag

第三优先级:
branch

辅助信息:
runtime-reported Python version
```

因此正确写法：

```text
Source:
6e7d571b831e6e4b82f1b2f8228cc84e9a0261a2
tag vllm-continuum-final-20260820

Runtime-reported:
0.1.dev10+g05f00f8a8
```

而不是只写：

```text
vLLM 0.1.dev10
```

---

# 第六部分：Git Archive / Backup

## 23. 项目 Git Bundle

为了避免 AutoDL 成为单点故障，项目已经创建 Git bundles。

目录：

```text
/root/autodl-tmp/vllm_project_archive_20260820/git-bundles/
```

包含：

```text
vllm-continuum.bundle
vllm-baseline.bundle
ucm-continuum.bundle
```

---

## 24. Bundle SHA256

历史归档中已创建：

```text
SHA256SUMS
```

历史记录中的 bundle SHA256 包括：

```text
ucm-continuum.bundle
dc1e0d1cefdbd7e7aed41ef6d5d305a4f9fcbcca409229e83c48c240e69f81e0

vllm-baseline.bundle
0b7d4e998e837f7030edcddad2ddadd7bb352c3198ac1a9672d75f4968c85bcf

vllm-continuum.bundle
4ea7bbae571db2e28169a0f6c5f01b4f009195edcc5cf06def9c795061b4b7bb
```

如果后续重新生成 bundles：

> 以新生成的 `SHA256SUMS` 为准。

---

## 25. Baseline Restore 验证

历史 baseline bundle 曾实际执行：

```text
git clone vllm-baseline.bundle
```

随后重新应用：

```text
baseline_worktree.patch
```

并与原工作树：

```text
vllm/v1/core/estimate_with_func.py
```

进行 `cmp`。

验证结果：

```text
BASELINE_RESTORE=PASS
```

说明 baseline 历史复现包可恢复。

---

# 第七部分：Project Review Snapshot

## 26. Review Snapshot

项目另有阶段性快照：

```text
/root/autodl-tmp/project_review_snapshot_20260820
```

以及：

```text
/root/autodl-tmp/project_review_snapshot_20260820.tar.gz
```

SHA256：

```text
f367f4fe6898cfac8c76fcba783a4eee798eebbc313bd76cdc3731938d7ccd53
```

该快照用于：

- 阶段评审；
- final_freeze；
- testsets；
- reference results；
- campaign configs；
- manifest。

它不是最终 clean deployment 的替代物。

---

# 第八部分：Compatibility Audit

## 27. Compatibility Audit 证据目录

项目曾建立：

```text
/root/autodl-tmp/vllm_compat_audit_20260820
```

其中按四部分保存原始调查证据：

```text
A_vllm_base/
B_continuum/
C_ucm/
D_joint/
```

这些原始文件记录：

- vLLM generated runtime 逻辑；
- Continuum 历史固定 TTL；
- Dynamic TTL 来源；
- UCM Git history；
- DRAMStore；
- UCMConnector；
- runtime context；
- empty-transfer fast path；
- joint integration。

正式发布文档可以引用这些事实，但：

> 不应直接使用此前可能被 terminal/heredoc 污染的自动生成 summary 作为正式最终文档。

当前：

```text
docs/COMPATIBILITY.md
docs/SOURCE_PROVENANCE.md
```

才是 clean deployment 下重新整理后的正式说明。

---

# 第九部分：Deployment Source of Truth

## 28. 三种“真相来源”必须区分

项目中同时存在：

```text
1. Git Source
2. Golden Runtime
3. Runtime Environment
```

三者用途不同。

### Git Source

回答：

> 我们最终改了什么代码？

位置：

```text
source/
```

### Golden Runtime

回答：

> 当前已验证机器实际加载的源码和 native artifact 是什么？

位置：

```text
deployment/runtime/golden/
```

### Runtime Environment

回答：

> 服务启动时实际生效了哪些环境变量和运行路径？

由 Smoke Test 保存：

```text
runtime_environment_effective.txt
full_system_config_declared.env
server.log
kv_transfer_config.json
```

---

## 29. Smoke Test 对 Provenance 的验证

最终 automated smoke 不只验证 API。

还验证：

```text
PYTHONPATH
```

实际指向：

```text
deployment/runtime/golden/unified-cache-management
deployment/runtime/golden/vllm-continuum
```

并保存：

```text
runtime_environment_effective.txt
```

因此可以证明：

> 最终 smoke 不是偷偷 import 原 `/root/autodl-tmp/vllm_workspace/src/...` 工作树，而是从 clean deployment Golden Runtime 启动。

---

# 第十部分：GitHub 发布建议

## 30. 推荐仓库划分

未来建议分为三个仓库。

### Repo 1

```text
vllm-continuum
```

内容：

> 最终 vLLM / Continuum 修改源码。

发布基准：

```text
6e7d571b831e6e4b82f1b2f8228cc84e9a0261a2
```

### Repo 2

```text
ucm-continuum
```

内容：

> 最终 UCM compatibility + system extensions。

发布基准：

```text
138769e7a13fc11b690bc6b954a31764785fb7c5
```

### Repo 3

```text
vllm-prefix-experiment
```

内容：

```text
deployment
benchmark
docs
configs
small workloads
aggregated results
manifests
```

---

## 31. GitHub 不应包含

不要直接上传：

```text
venv
model weights
Golden Runtime 大型 native binary（除非 Release 专门提供）
raw EnvBench 2 GiB 数据
SSD backing files
所有历史 raw logs
AutoDL cache
```

---

## 32. 推荐发布身份

如果以后制作正式 release，建议：

```text
Project release:
v1.0

vLLM / Continuum source:
vllm-continuum-final-20260820

UCM source:
ucm-continuum-final-20260820
```

并在 release manifest 中记录：

```text
commit
tag
source SHA256
runtime SHA256
model
environment
benchmark configuration
```

---

# 第十一部分：Final Source Contract

## 33. 最终源码契约

从当前 clean deployment 开始，以下内容视为正式冻结：

```text
source/vllm-continuum
source/unified-cache-management

deployment/runtime/golden
deployment/configs/full_system.env

deployment/scripts/start_full.sh
deployment/scripts/stop.sh
deployment/scripts/status.sh
deployment/scripts/smoke_test.sh
```

只有在正式 benchmark 发现真实系统 bug 时才重新修改。

如果修改：

> 必须生成新的 Git commit / tag，并重新生成 source/runtime manifests。

---

## 34. 当前最终版本总结

### vLLM / Continuum

```text
branch:
joint-offload-v1

commit:
6e7d571b831e6e4b82f1b2f8228cc84e9a0261a2

tag:
vllm-continuum-final-20260820
```

### UCM

```text
branch:
joint-offload-v1

commit:
be59181f50e515496a7f19a38178c8c2a05cf251

tag:
ucm-continuum-final-20260822
```

### Runtime

```text
Python:
3.12.3

vLLM runtime-reported:
0.1.dev10+g05f00f8a8

GPU:
RTX 4090

Model:
Qwen3-0.6B
```

### Clean Source

```text
vLLM files:
3233

UCM files:
438
```

### Exact Runtime

```text
Golden Source Integrity:
PASS

Golden Native Import:
PASS

Full System Smoke:
PASS
```

---

## 35. 结论

本项目最终源码应理解为：

```text
不是：
某个官方 release 的原样副本

而是：
可追溯到明确 Git commit/tag 的项目冻结版本
```

其中：

```text
Git commit/tag
→ 定义源码身份

Clean Source
→ 定义可审查代码

Golden Runtime
→ 定义已验证二进制运行快照

Smoke Test evidence
→ 定义实际生效部署状态
```

最终 provenance 链：

```text
Historical vLLM / Continuum / UCM
            ↓
Project Compatibility & Integration
            ↓
Frozen Git Commits / Tags
            ↓
git archive Clean Source
            ↓
Golden Runtime
            ↓
Automated Smoke Test
            ↓
Final Benchmark / Release
```

这条链路保证最终系统的源码、运行状态与实验结果都能够被明确追溯。

---

## Final UCM Freeze Update — 2026-08-22

The earlier UCM checkpoint:

```text
138769e7a13fc11b690bc6b954a31764785fb7c5
ucm-continuum-final-20260820
```

is retained in this document where it describes historical evolution, especially the
empty-transfer optimization checkpoint.

The final UCM source identity used by deployment and the formal Full EnvBench campaign is:

```text
branch = joint-offload-v1
HEAD   = be59181f50e515496a7f19a38178c8c2a05cf251
tag    = ucm-continuum-final-20260822
```

Final commit subject:

```text
fix: initialize KV caches before empty-load fast path
```

This later freeze incorporates the empty-load lifecycle fix validated before the formal
Full EnvBench V/U/C/F campaign.

Formal benchmark provenance:

```text
Full Runtime-Eligible EnvBench
1274 trajectories
13419 tool events
14693 requests per case
9 shards
V/U/C/F
36/36 case PASS
58772 total requests across four cases
```

The canonical final performance record is `docs/BENCHMARK.md` and
`benchmark/results/full_envbench_final_20260823/`.

