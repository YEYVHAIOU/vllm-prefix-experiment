# 部署文档（DEPLOYMENT）

## 1. 文档目的

本文档说明本项目最终系统在已验证 AutoDL 环境中的部署方式、运行依赖、配置参数、启动与停止流程、自动 Smoke Test 验收方式，以及在其他机器或软件栈上重建时需要注意的兼容性边界。

本项目最终系统不是“直接克隆上游 vLLM / Continuum / UCM 后即可得到的原始开源版本”，而是经过以下演进与兼容性修改后的联合系统：

- vLLM / Continuum：加入并优化 Dynamic TTL；
- vLLM Runtime Context：向 UCM 传递 GPU pressure、temporal hints、上下文规模、finish probability、prefill / reload cost、terminal marker 等信息；
- UCM WHEN：`joint + cost_full`；
- UCM WHAT：`frontier_tail`，保留块数 `K=4`；
- UCM WHERE：`UcmTieredStore`，DRAM-first、local SSD-second；
- 针对当前 vLLM / UCM 组合补充了兼容性与性能修正，包括空传输 fast path、联合 offload decision、TieredStore、Frontier-Tail partial retention 等。

最终架构可概括为：

> Dynamic TTL + cost_full WHEN + Frontier-Tail WHAT (K=4) + TieredStore WHERE

其中 Frontier-Tail 是外部 KV backing 的部分保留策略，不应表述为“UCM Sparse Attention”。

---

## 2. 已验证部署范围

本项目已经在以下真实环境中完成从交付目录启动、健康检查、真实推理、Metrics 读取、策略路径验证和正常停止：

- GPU：NVIDIA GeForce RTX 4090，24564 MiB；
- Driver：595.71.05；
- Python：3.12.3；
- PyTorch：2.8.0+cu129；
- PyTorch CUDA：12.9；
- CUDA Toolkit / `CUDA_HOME`：`/usr/local/cuda-12.8`；
- `nvcc`：CUDA 12.8，V12.8.93；
- GCC：11.2；
- Transformers：4.56.2；
- Tokenizers：0.22.2；
- Hugging Face Hub：0.36.2；
- 模型：Qwen3-0.6B；
- `max_model_len=4096`；
- vLLM runtime-reported version：`0.1.dev10+g05f00f8a8`。

注意：

1. runtime-reported vLLM version 是安装 / 构建阶段生成的 `_version.py` 身份，不是最终源码版本的唯一身份；
2. 最终源码身份应以 Git commit / tag 为准：
   - vLLM / Continuum：`6e7d571b831e6e4b82f1b2f8228cc84e9a0261a2`
   - tag：`vllm-continuum-final-20260820`
   - UCM：`be59181f50e515496a7f19a38178c8c2a05cf251`
   - tag：`ucm-continuum-final-20260822`

---

## 3. 交付目录结构

最终 clean deployment 根目录：

```text
vllm_deploy_clean_20260820/
├── source/
│   ├── vllm-continuum/
│   └── unified-cache-management/
├── deployment/
│   ├── configs/
│   │   └── full_system.env
│   ├── scripts/
│   │   ├── start_full.sh
│   │   ├── stop.sh
│   │   ├── status.sh
│   │   └── smoke_test.sh
│   └── runtime/
│       ├── golden/
│       │   ├── vllm-continuum/
│       │   └── unified-cache-management/
│       ├── runs/
│       ├── state/
│       └── ssd_store/
├── benchmark/
│   ├── workloads/
│   ├── scripts/
│   └── results/
├── docs/
│   ├── DEPLOYMENT.md
│   ├── COMPATIBILITY.md
│   ├── BENCHMARK.md
│   └── SOURCE_PROVENANCE.md
├── manifests/
└── README.md
```

### 3.1 Frozen Release 中的 `source/`

本节描述完整冻结 release 中的源码快照。GitHub 精简版不再重复包含 `source/`，而是使用两个独立源码仓库：

- <https://github.com/YEYVHAIOU/vllm-continuum>
- <https://github.com/YEYVHAIOU/unified-cache-management-continuum>

冻结 release 中的 `source/` 是纯 Git tracked source 交付：

- `source/vllm-continuum/`：3233 个源码文件；
- `source/unified-cache-management/`：438 个源码文件；
- 不包含 `.so`、`.pyc`、`__pycache__` 和生成的 `_version.py`；
- 适合代码审查、GitHub 发布、源码重建和版本追溯。

### 3.2 Frozen Release 中的 `deployment/runtime/golden/`

Golden Runtime 是“已验证运行树”，由纯源码副本加当前机器 / Python / CUDA 栈所需生成产物构成。

相对于 `source/`，Golden Runtime 仅增加运行必要的 generated / native artifacts：

vLLM 部分包括：

- `vllm/_C.abi3.so`
- `vllm/_flashmla_C.abi3.so`
- `vllm/_moe_C.abi3.so`
- `vllm/cumem_allocator.abi3.so`
- `vllm/_version.py`
- `vllm/vllm_flash_attn/` 下生成的 Python 与 native extension

UCM 部分包括：

- `ucm/store/nfsstore/ucmnfsstore.cpython-312-x86_64-linux-gnu.so`
- `ucm/store/pcstore/ucmpcstore.cpython-312-x86_64-linux-gnu.so`

Golden Runtime 与纯源码已经完成一致性检查：

- pure source missing：0；
- changed source files：0；
- runtime extra files：16。

因此：

> Golden Runtime = 原始 clean source + 当前已验证软件栈所需 generated/native runtime artifacts。

---

## 4. Exact AutoDL 依赖路径

当前已验证 exact deployment 使用：

```text
VENV_PATH=/root/autodl-tmp/vllm_workspace/envs/vllm-continuum
MODEL_PATH=/root/autodl-tmp/vllm_workspace/models/Qwen3-0.6B
CUDA_HOME=/usr/local/cuda-12.8
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

其中 `LD_PRELOAD` 是 UCM native store 的必要兼容项。未设置时，当前 AutoDL 环境中的 Miniconda `libstdc++.so.6` 会导致：

```text
GLIBCXX_3.4.30 not found
```

因此该变量已经固定写入 `deployment/configs/full_system.env`，不依赖登录 shell 的临时环境。

---

## 5. 最终 Full System 配置

配置文件：

```text
deployment/configs/full_system.env
```

### 5.1 API / vLLM

| 参数 | 最终值 |
|---|---:|
| Host | `127.0.0.1` |
| Port | `8000` |
| Model | `Qwen/Qwen3-0.6B` |
| Max model length | `4096` |
| GPU memory utilization | `0.80` |
| Swap space | `1 GiB` |
| Prefix caching | Enabled |
| Scheduler | `continuum` |
| Enforce eager | Disabled |

最终正式部署使用 `gpu_memory_utilization=0.80` 且不启用 `--enforce-eager`。这是项目后期 fidelity 配置，不是临时选择。

### 5.2 Continuum / Dynamic TTL

| 参数 | 最终值 |
|---|---:|
| `CONTINUUM_TTL_CDF_IMPL` | `optimized` |
| `CONTINUUM_HISTORY_THRESHOLD` | `5` |
| `CONTINUUM_DEFAULT_TTL_SECONDS` | `2.0` |
| `CONTINUUM_PREFILL_PROFILE_SCALE` | `1.0` |
| `CONTINUUM_TTL_TIMING_INTERVAL` | `0` |

当前 prefill profile 点为：

```text
0    -> 0
512  -> 0.016206744
1024 -> 0.038612129
2048 -> 0.095269512
3072 -> 0.164592375
4000 -> 0.234015222
```

该基础 profile 历史上来自 RTX 4060 测量。迁移到 RTX 4090 后做过 scale / counterfactual sensitivity 分析，最终仍采用 `scale=1.0`；但没有单独重新测量并拟合一条完整的 4090 prefill curve。该限制应在兼容性文档中保留说明。

### 5.3 UCM WHEN

| 参数 | 最终值 |
|---|---:|
| `UCM_OFFLOAD_POLICY` | `joint` |
| `UCM_JOINT_DECISION` | `cost_full` |
| `UCM_PRESSURE_THRESHOLD` | `0.65` |
| `UCM_JOINT_HIGH_THRESHOLD` | `0.90` |
| `UCM_JOINT_AFTER_TTL_THRESHOLD` | `0.50` |

传输成本模型：

```text
dump_cost_ms = 2.8000 + 1.5005 * n_blocks
load_cost_ms = 0.3510 + 0.3345 * n_blocks
```

### 5.4 UCM WHAT

| 参数 | 最终值 |
|---|---:|
| `UCM_WHAT_POLICY` | `frontier_tail` |
| `UCM_RETAIN_BLOCKS` | `4` |

语义：

> 当联合决策认为外部 KV backing 有价值时，不默认全量保留所有候选 KV，而是采用 Frontier-Tail Partial Retention，保留 frontier-tail 范围中的有限块，当前固定 `K=4`。

### 5.5 UCM WHERE / TieredStore

| 参数 | 最终值 |
|---|---:|
| Store | `UcmTieredStore` |
| DRAM budget | `128 MiB` |
| SSD root | `deployment/runtime/ssd_store` |
| `use_direct` | `false` |
| transfer stream number | `2` |
| transfer buffer number | `64` |

实际启动日志已经验证 transfer-enabled NFSStore data path 使用：

```text
TransferEnable = true
DeviceId = 0
StreamNumber = 2
BufferNumber = 64
```

另一个 `TransferEnable=false` 的内部 NFSStore 实例会保留底层默认：

```text
StreamNumber = 32
BufferNumber = 512
```

这不表示最终参数未生效；2 / 64 已经在真实 transfer-enabled path 上被确认使用。

### 5.6 Reproducibility

```text
PYTHONHASHSEED=123456
PYTHONDONTWRITEBYTECODE=1
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
VLLM_LOGGING_LEVEL=INFO
UCM_TRANSFER_PROFILE=0
```

注意：

- `UCM_HASH_SEED` 不是环境变量，不应加入配置；
- `UCM_EMPTY_TRANSFER_FASTPATH_V1` 也不是环境变量，它是代码路径 / 标记，不应导出到 shell。

---

## 6. 启动服务

GitHub 精简版默认采用三个仓库并列的目录结构，并要求显式指定本机 Python 环境和模型目录：

```text
workspace/
├── vllm-prefix-experiment/
├── vllm-continuum/
└── unified-cache-management-continuum/
```

进入主仓库后：

```bash
export VENV_PATH=/path/to/your/vllm-environment
export MODEL_PATH=/path/to/Qwen3-0.6B
bash deployment/scripts/start_full.sh
```

如源码仓库不在同级目录，可额外设置 `VLLM_REPO` 和 `UCM_REPO`。

脚本会：

1. 加载 `full_system.env`；
2. 默认使用同级目录中的 `../vllm-continuum` 和 `../unified-cache-management-continuum`，也可通过 `VLLM_REPO` / `UCM_REPO` 覆盖；
3. 检查 Python、vLLM executable、模型、源码树和端口；
4. 创建 runtime state、run directory 和 SSD store；
5. 生成 `kv_transfer_config.json`；
6. 使用 `PYTHONPATH=$UCM_REPO:$VLLM_REPO` 启动服务；
7. 保存 PID 和运行目录。

典型输出：

```text
Starting Full System
  model:       /root/autodl-tmp/vllm_workspace/models/Qwen3-0.6B
  scheduler:   continuum
  GPU util:    0.80
  eager:       disabled
  WHEN:        joint/cost_full
  WHAT:        frontier_tail K=4
  WHERE:       TieredStore DRAM=128MiB
  SSD:         .../deployment/runtime/ssd_store
  run:         .../deployment/runtime/runs/full_YYYYMMDD_HHMMSS
PID=...
SERVER_LOG=.../server.log
```

---

## 7. 查看状态

```bash
./deployment/scripts/status.sh
```

运行中：

```text
STATUS=RUNNING PID=<pid>
RUN_DIR=<path>
```

停止后：

```text
STATUS=STOPPED
```

---

## 8. 健康检查

服务启动后：

```bash
curl -fsS http://127.0.0.1:8000/health
```

若返回成功状态码，则 API server 已启动。

已验证 clean deployment 的 `/health` 返回 HTTP 200。

---

## 9. 自动 Smoke Test

推荐使用：

```bash
./deployment/scripts/smoke_test.sh
```

该脚本会自动完成：

1. 检查当前无已运行 server；
2. 调用 `start_full.sh`；
3. 等待 `/health`；
4. 从 `/proc/<pid>/environ` 保存实际生效环境；
5. 验证 Golden Runtime `PYTHONPATH`；
6. 验证 Dynamic TTL / UCM 核心环境变量；
7. 生成 `smoke_request.json`；
8. 调用 `/v1/chat/completions`；
9. 验证 HTTP 200 和 OpenAI-compatible response；
10. 拉取 `/metrics`；
11. 验证真实 2xx chat request；
12. 从 `server.log` 验证 Continuum / UCM / TieredStore 参数；
13. 检查 `Traceback / ImportError / GLIBCXX / EngineCore failed`；
14. 停止服务；
15. 验证 `STATUS=STOPPED`；
16. 验证端口释放；
17. 生成 `smoke_summary.txt`。

最终已验证结果：

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

因此当前 clean delivery tree 已经真实完成：

> 启动 → health → 推理 → metrics → 策略配置验证 → 无运行时异常 → 停止 → 端口释放。

---

## 10. 手工真实推理

若不使用 Smoke Test，也可以直接调用 OpenAI-compatible API：

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {
        "role": "user",
        "content": "Hello"
      }
    ],
    "temperature": 0,
    "max_tokens": 32
  }'
```

注意：Qwen3 默认可能产生 `<think>` 内容，并可能因较小的 `max_tokens` 以 `finish_reason=length` 结束。这不表示部署失败。

部署 Smoke Test 验证的是完整 inference/API 链路是否正常，而不是要求模型逐字返回某个固定字符串。

---

## 11. 查看 Metrics

```bash
curl -fsS http://127.0.0.1:8000/metrics
```

当前已验证可看到：

- HTTP request counter；
- prompt / generation throughput；
- request latency；
- KV cache config；
- prefix caching enabled；
- GPU KV cache blocks；
- chat completions 2xx counter。

单请求 Smoke Test 不用于证明高压力 offload 行为。

Dynamic TTL、UCM external dump/load、DRAM/SSD activity 和性能差异由两级实验共同验证：C76/C96/C128 高压力 EnvBench-derived benchmark 用于容量/压力验证，最终 V/U/C/F 主结果使用 Full Runtime-Eligible EnvBench（C128）验证。

---

## 12. 停止服务

```bash
./deployment/scripts/stop.sh
```

成功：

```text
SERVER_STOPPED
```

再次查看：

```bash
./deployment/scripts/status.sh
```

应为：

```text
STATUS=STOPPED
```

---

## 13. 运行日志与证据

每次启动会创建：

```text
deployment/runtime/runs/full_YYYYMMDD_HHMMSS/
```

典型文件包括：

```text
server.log
kv_transfer_config.json
runtime_environment_effective.txt
full_system_config_declared.env
smoke_request.json
smoke_response.json
smoke_metrics.txt
smoke_runtime_errors.txt
post_stop_status.txt
smoke_summary.txt
```

这些文件可作为部署验收和问题排查证据。

---

## 14. 已验证关键日志

成功启动时应至少能观察到：

```text
scheduling_policy = continuum
enable_prefix_caching = True
gpu_memory_utilization = 0.8
```

UCM：

```text
UCM offload policy=joint
UCM WHAT policy=frontier_tail ... retain_blocks=4
UCM joint decision=cost_full
UCM tiered store initialized
```

Continuum：

```text
Continuum history threshold=5
Continuum default TTL=2.000000
Continuum prefill profile scale=1.000
```

真实请求路径可出现：

```text
UCM_OFFLOAD_DECISION policy=joint
UCM_WHAT_DECISION policy=frontier_tail
```

在单请求、极低 GPU pressure 下：

```text
selected=0
reason=joint_cost_full_zero_evict_skip
```

是正常现象，不表示 UCM 未启用。

---

## 15. UCM 版本兼容性说明

当前 UCM upstream patch whitelist 显式声明支持：

```text
vLLM 0.9.2
```

当前项目 runtime-reported vLLM version 为：

```text
0.1.dev10+g05f00f8a8
```

启动 / import 时会产生 unsupported-version warning，但当前项目兼容层实际输出：

```text
All vLLM patches applied successfully for version 0.1.dev10+g05f00f8a8
```

并已经完成：

- import；
- native extension；
- TieredStore；
- API startup；
- real inference；
- UCM decision path；
- smoke test；

的实际运行验证。

因此应表述为：

> 当前组合不属于 UCM upstream whitelist 中的原生支持版本，而是经过本项目兼容性适配并完成实机验证的联合版本。

不能表述为“UCM 官方原生支持当前 vLLM 版本”。

详细修改见 `COMPATIBILITY.md`。

---

## 16. 三种部署级别

### Level A：Exact Verified Deployment

目标：复现当前 AutoDL / RTX 4090 已验证环境。

使用：

- clean delivery tree；
- Golden Runtime；
- 当前 verified venv；
- 当前模型路径；
- CUDA 12.8 Toolkit；
- `LD_PRELOAD` 指定系统 libstdc++。

这是本项目当前最强复现保证。

### Level B：Compatible Source Rebuild

目标：在相近 Linux / NVIDIA / Python / CUDA 栈上重新构建。

使用：

```text
source/vllm-continuum/
source/unified-cache-management/
```

重新生成：

- vLLM native extensions；
- vLLM FlashAttention artifacts；
- `_version.py`；
- UCM NFSStore / PCStore native extensions。

随后继续使用同一：

```text
full_system.env
start_full.sh
smoke_test.sh
```

但需要把 `VENV_PATH`、`MODEL_PATH` 等机器路径改为目标环境。

### Level C：Best-effort Porting

当 Python、CUDA、PyTorch、GPU 架构或 vLLM/UCM ABI 变化较大时，不保证 Golden Runtime `.so` 可直接复用。

此时应：

1. 从 pure source 重建；
2. 验证 UCM patch compatibility；
3. 检查 native ABI；
4. 检查 `libstdc++`；
5. 重新执行 Smoke Test；
6. 再执行正式 benchmark。

---

## 17. 不应直接复制 Golden Native Artifacts 的情况

以下变化可能使 Golden `.so` 不再兼容：

- Python ABI 变化；
- CUDA major/minor stack 明显变化；
- PyTorch ABI 变化；
- GPU compute capability 变化；
- GCC / GLIBCXX ABI 变化；
- vLLM native source 变化；
- UCM native source 变化。

因此：

> 在完整冻结 release 中，`source/` 是长期可维护源码快照，`deployment/runtime/golden/` 是当时已验证的二进制运行快照。GitHub 精简版不提交 Golden Runtime，而是使用两个独立源码仓库。

冻结交付与 GitHub 可复现仓库用途不同。

---

## 18. 模型与数据集不纳入源码目录

原 AutoDL 最终验证环境中的模型路径：

```text
/root/autodl-tmp/vllm_workspace/models/Qwen3-0.6B
```

原 AutoDL 最终验证环境中的 raw EnvBench 路径：

```text
/root/autodl-tmp/vllm_workspace/datasets/envbench_trajectories_raw
```

模型约 1.5 GiB，raw EnvBench 约 2.0 GiB。

这些大资产不纳入 GitHub 精简仓库；完整冻结 release 中也与源码快照保持分离。

GitHub 部署时通过环境变量 `MODEL_PATH` 指向本地模型目录。

压力/容量 benchmark 使用的已冻结 workload：

```text
benchmark/workloads/envbench_high_contention_balanced_min3500.csv
```

最终 Full EnvBench 正式 V/U/C/F 主评测使用：

```text
benchmark/workloads/envbench_full_runtime_eligible_max4000.csv
benchmark/workloads/full_shards_9/
benchmark/results/full_envbench_final_20260823/
```

其中 min3500 是 pressure-enhanced workload，不代表原始 prompt 均自然达到 3500 tokens；
最终 Full workload 保留完整 runtime-eligible trajectories，并在 4096-token server 配置下
要求所有事件 input_tokens <= 4000。

---

## 19. 常见问题排查

### 19.1 `GLIBCXX_3.4.30 not found`

确认：

```bash
grep '^LD_PRELOAD=' deployment/configs/full_system.env
```

应为：

```text
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

### 19.2 UCM unsupported vLLM warning

如果同时出现：

```text
All vLLM patches applied successfully
```

且 Smoke Test 通过，则该 warning 是 upstream whitelist mismatch 的兼容性提示。

如果 patch apply 失败，则不能忽略，应停止部署并检查版本接口差异。

### 19.3 8000 端口占用

`start_full.sh` 会主动检查端口。

当前精简 AutoDL 镜像可能没有 `ss`，可用 Python 检查：

```bash
python -c 'import socket; s=socket.socket(); s.settimeout(.3); r=s.connect_ex(("127.0.0.1",8000)); s.close(); print("occupied" if r==0 else "free")'
```

### 19.4 Smoke Test 中模型没有输出指定短语

只要：

```text
SMOKE_HTTP=200
SMOKE_RESPONSE=PASS
```

且 response 存在有效 choice/content，则 inference 链路通过。

Qwen3 可能先输出 reasoning，并因 `max_tokens=32` 截断。

### 19.5 单请求没有 external dump/load

正常。

单请求压力极低，不用于证明 offload 性能。

正式算法与系统行为应运行高并发 EnvBench-derived workload。

---

## 20. 部署验收标准

部署可判定为通过，至少需要：

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

同时源码 / runtime manifests 应通过 SHA256 校验。

---

## 21. 与正式 Benchmark 的边界

Deployment Smoke Test 只回答：

> 系统能否从交付目录正确启动并完成真实推理？

它不回答：

> vLLM、vLLM+UCM、vLLM+Continuum、vLLM+UCM+Continuum 的性能差异是多少？

正式性能评测必须使用统一 workload、统一模型、统一 GPU、统一 GPU memory utilization、统一并发和统一 runner。

本项目最终正式比较定义为：

```text
V = FCFS + UCM OFF
U = FCFS + UCM ON
C = Continuum + UCM OFF
F = Continuum + UCM ON
```

详细方法与结果见 `BENCHMARK.md`。

---


## 21.1 Full EnvBench 最终闭环结果

最终正式主评测采用 `Full Runtime-Eligible EnvBench`：

- 1274 trajectories；
- 13419 tool events；
- 每种 case 14693 requests；
- 9 个 deterministic shards；
- concurrency = 128；
- reset-aware prefix replay = ON；
- 四组 V/U/C/F 共 36 个 case；
- 共执行 58772 requests；
- 所有请求成功，36/36 case PASS。

最终聚合结果：

| Case | Req/s | Prefix Hit | Mean TTFT | Mean E2E | Mean JCT |
|---|---:|---:|---:|---:|---:|
| V | 79.96 | 65.17% | 0.844 s | 0.976 s | 11.467 s |
| U | 25.86 | 67.16% | 2.793 s | 3.808 s | 44.142 s |
| C | 116.72 | 86.90% | 0.302 s | 0.490 s | 5.873 s |
| F | 109.24 | 86.89% | 0.327 s | 0.529 s | 6.325 s |

Full System（F）GPU KV Cache 利用率：

- mean = 28.44%；
- P95 = 61.03%；
- P99 = 65.31%；
- max = 71.69%。

关键结论：

1. Continuum 是当前正式实验中的主要性能收益来源；
2. C 相比 V：request throughput +45.97%，mean E2E -49.78%，Prefix Hit +21.73 pp；
3. eager/full UCM（U）产生大量 SSD migration，平均约 38.17 GiB/shard，性能显著下降；
4. Full System 的 `joint/cost_full` 在 Full EnvBench 中最终 selected blocks = 0，
   成功避免了 U 中不划算的 external KV migration；
5. F 相比 C 仍有约 6.40% throughput overhead，因此不能声称当前 UCM 集成进一步提升了
   Continuum 性能；
6. Frontier-Tail 已完成集成，但由于正式 Full workload 下 cost-aware WHEN 未选择实际外迁，
   不能声称 Frontier-Tail SSD I/O 在该实验中带来了正收益。

完整 workload 定义、reset-aware replay、聚合规则、尾延迟、KV/TTL/UCM/SSD 结果与报告边界，
以 `docs/BENCHMARK.md` 为准。


## 22. 结论

当前部署闭环已经实机完成：

```text
Clean Source
    +
Golden Runtime
    +
Declarative Full-System Config
    +
Start / Stop / Status Scripts
    +
Automated Smoke Test
    ↓
RTX 4090 AutoDL 实际部署验证通过
```

因此，本项目已经具备“源码 + 可验证部署流程 + 运行配置 + 自动验收”的完整部署交付基础。
