# 部署指南

本文档说明 vLLM + Continuum + UCM 联合系统的部署方式、运行依赖、启动与停止流程，以及在新机器上重新构建时需要注意的兼容性条件。

GitHub 版本采用三个并列仓库：主实验仓库负责部署脚本、基准测试和文档；两个源码仓库分别保存修改后的 vLLM/Continuum 与 UCM。完整冻结 release 另外保留 Golden Runtime、运行日志和历史证据，用于精确追溯原 AutoDL 环境，但这些二进制运行快照不作为 GitHub 常规部署依赖。

## 已验证环境

最终正式实验和部署验收运行于以下环境：

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090，24564 MiB |
| CPU | 16 vCPU |
| 主机内存 | 约 120 GB |
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu129 |
| PyTorch CUDA | 12.9 |
| CUDA Toolkit | 12.8 |
| `CUDA_HOME` | `/usr/local/cuda-12.8` |
| GCC | 11.2 |
| Transformers | 4.56.2 |
| Tokenizers | 0.22.2 |
| Hugging Face Hub | 0.36.2 |
| 模型 | Qwen3-0.6B |
| vLLM 运行时版本 | `0.1.dev10+g05f00f8a8` |

`vllm.__version__` 由构建/安装阶段生成的 `_version.py` 提供。最终源码版本以 Git 提交和标签为准：

```text
vLLM + Continuum
commit = 6e7d571b831e6e4b82f1b2f8228cc84e9a0261a2
tag    = vllm-continuum-final-20260820

UCM
commit = be59181f50e515496a7f19a38178c8c2a05cf251
tag    = ucm-continuum-final-20260822
```

## 获取源码

推荐目录结构：

```text
workspace/
├── vllm-prefix-experiment/
├── vllm-continuum/
└── unified-cache-management-continuum/
```

三个仓库保持同级目录即可。主实验仓库中的启动脚本默认从同级目录查找两个源码仓库。

克隆后，将源码切换到冻结版本：

```bash
cd vllm-continuum
git checkout vllm-continuum-final-20260820

cd ../unified-cache-management-continuum
git checkout ucm-continuum-final-20260822
```

如果源码仓库位于其他位置，可在启动前设置：

```bash
export VLLM_REPO=/path/to/vllm-continuum
export UCM_REPO=/path/to/unified-cache-management-continuum
```

## Python 环境与模型

GitHub 精简仓库不包含 Python 虚拟环境、模型权重和原始 EnvBench 数据。部署前至少需要准备：

```text
一个可运行当前 vLLM/UCM 源码的 Python 环境
Qwen3-0.6B 模型目录
CUDA / 编译工具链
```

启动脚本要求显式指定：

```bash
export VENV_PATH=/path/to/your/vllm-environment
export MODEL_PATH=/path/to/Qwen3-0.6B
```

在原 AutoDL 验证环境中使用：

```text
VENV_PATH=/root/autodl-tmp/vllm_workspace/envs/vllm-continuum
MODEL_PATH=/root/autodl-tmp/vllm_workspace/models/Qwen3-0.6B
CUDA_HOME=/usr/local/cuda-12.8
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

这些 AutoDL 路径只是原正式实验的实际环境记录；GitHub 部署不要求复用相同绝对路径。

## 原生扩展与构建

vLLM 和 UCM 均包含原生扩展，因此纯源码仓库不能保证在任意 Python / CUDA / PyTorch 组合下直接复用原 AutoDL 的二进制文件。

完整冻结 release 中额外保存的 Golden Runtime 主要包含：

```text
vLLM:
  _C.abi3.so
  _flashmla_C.abi3.so
  _moe_C.abi3.so
  cumem_allocator.abi3.so
  vllm_flash_attn/*
  _version.py

UCM:
  ucmnfsstore.cpython-312-x86_64-linux-gnu.so
  ucmpcstore.cpython-312-x86_64-linux-gnu.so
```

这些文件与 Python ABI、PyTorch、CUDA Toolkit、GCC/GLIBCXX 和 GPU 架构存在绑定关系。新机器部署应优先从最终源码重新构建原生组件，而不是复制冻结 release 中的 `.so`。

当前 AutoDL 环境曾出现：

```text
GLIBCXX_3.4.30 not found
```

原因是 Conda 环境中的 `libstdc++.so.6` 与 UCM 原生扩展所需 ABI 不匹配。原验证环境通过：

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

显式使用系统 `libstdc++`。新机器只有在遇到相同 ABI 问题时才需要采用对应处理。

## 完整系统配置

正式配置文件为：

```text
deployment/configs/full_system.env
```

### vLLM 与服务参数

| 参数 | 最终值 |
|---|---:|
| Host | `127.0.0.1` |
| Port | `8000` |
| Model | `Qwen/Qwen3-0.6B` |
| `max_model_len` | `4096` |
| GPU 显存利用率 | `0.80` |
| Swap | `1 GiB` |
| Prefix Cache | 开启 |
| 调度器 | `continuum` |
| `enforce_eager` | 关闭 |

`max_model_len=4096` 是本项目服务实验使用的限制，不代表模型固有最大上下文长度。

### Continuum / Dynamic TTL

| 参数 | 最终值 |
|---|---:|
| `CONTINUUM_TTL_CDF_IMPL` | `optimized` |
| `CONTINUUM_HISTORY_THRESHOLD` | `5` |
| `CONTINUUM_DEFAULT_TTL_SECONDS` | `2.0` |
| `CONTINUUM_PREFILL_PROFILE_SCALE` | `1.0` |
| `CONTINUUM_TTL_TIMING_INTERVAL` | `0` |

当前 Prefill 曲线为：

```text
0    -> 0
512  -> 0.016206744
1024 -> 0.038612129
2048 -> 0.095269512
3072 -> 0.164592375
4000 -> 0.234015222
```

基础曲线历史上来自 RTX 4060。迁移到 RTX 4090 后完成过缩放敏感性实验，但没有重新完整拟合一条 4090 曲线，因此跨 GPU 复现时建议重新校准。

### UCM WHEN

| 参数 | 最终值 |
|---|---:|
| `UCM_OFFLOAD_POLICY` | `joint` |
| `UCM_JOINT_DECISION` | `cost_full` |
| `UCM_PRESSURE_THRESHOLD` | `0.65` |
| `UCM_JOINT_HIGH_THRESHOLD` | `0.90` |
| `UCM_JOINT_AFTER_TTL_THRESHOLD` | `0.50` |

当前传输成本模型：

```text
dump_cost_ms = 2.8000 + 1.5005 * n_blocks
load_cost_ms = 0.3510 + 0.3345 * n_blocks
```

### UCM WHAT

| 参数 | 最终值 |
|---|---:|
| `UCM_WHAT_POLICY` | `frontier_tail` |
| `UCM_RETAIN_BLOCKS` | `4` |

Frontier-Tail 用于控制跨存储层级保留的 KV blocks 数量。它属于外部 KV 部分保留策略，不修改 Attention 计算本身。

### UCM WHERE

| 参数 | 最终值 |
|---|---:|
| Store | `UcmTieredStore` |
| DRAM 容量 | `128 MiB` |
| SSD 目录 | `deployment/runtime/ssd_store` |
| Stream 数 | `2` |
| Buffer 数 | `64` |
| `use_direct` | `false` |

当前放置顺序为 DRAM-first / SSD-second。

## 启动服务

进入主实验仓库：

```bash
cd vllm-prefix-experiment
```

设置本机环境和模型路径：

```bash
export VENV_PATH=/path/to/your/vllm-environment
export MODEL_PATH=/path/to/Qwen3-0.6B
```

如果需要，可覆盖源码目录和 CUDA 设置：

```bash
export VLLM_REPO=/path/to/vllm-continuum
export UCM_REPO=/path/to/unified-cache-management-continuum
export CUDA_HOME=/usr/local/cuda-12.8
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

启动：

```bash
bash deployment/scripts/start_full.sh
```

脚本会加载 `full_system.env`，检查 Python、模型、源码树与端口，创建运行目录和 SSD store，生成 KV transfer 配置，并使用：

```text
PYTHONPATH=$UCM_REPO:$VLLM_REPO
```

启动服务。

运行状态保存在：

```text
deployment/runtime/state/
```

每次启动产生独立运行目录：

```text
deployment/runtime/runs/full_YYYYMMDD_HHMMSS/
```

## 查看状态与健康检查

查看服务状态：

```bash
bash deployment/scripts/status.sh
```

运行中会输出类似：

```text
STATUS=RUNNING PID=<pid>
RUN_DIR=<path>
```

HTTP 健康检查：

```bash
curl -fsS http://127.0.0.1:8000/health
```

原验证环境已确认 `/health` 返回 HTTP 200。

## 自动 Smoke Test

推荐使用：

```bash
bash deployment/scripts/smoke_test.sh
```

Smoke Test 会自动完成服务启动、健康检查、环境变量核验、真实 OpenAI-compatible 请求、指标读取、关键日志检查、服务停止和端口释放。

最终冻结环境中的验收结果：

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

该测试验证的是“部署链路能够完成真实启动和推理”，不是性能基准测试。

## 手工请求

服务启动后，可直接调用 OpenAI-compatible API：

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

Qwen3 可能先产生 reasoning 内容，较小的 `max_tokens` 也可能导致 `finish_reason=length`。部署验证只检查 API 与推理链路是否正常，不要求模型返回固定文本。

## Metrics

读取服务指标：

```bash
curl -fsS http://127.0.0.1:8000/metrics
```

正式部署已验证能够获得 HTTP 请求、吞吐、延迟、Prefix Cache 与 GPU KV Cache 相关指标。

单请求 Smoke Test 的 GPU 压力很低，因此可能出现：

```text
selected=0
reason=joint_cost_full_zero_evict_skip
```

这表示当前成本模型判断没有必要执行 KV 外部迁移，并不代表 UCM 未启用。

高压力下的 Dynamic TTL、UCM、DRAM/SSD 与 KV Cache 行为由基准测试负责验证，详见 `docs/BENCHMARK.md`。

## 停止服务

```bash
bash deployment/scripts/stop.sh
```

成功后：

```text
SERVER_STOPPED
```

再次执行：

```bash
bash deployment/scripts/status.sh
```

应返回：

```text
STATUS=STOPPED
```

## 运行产物

每次运行目录通常包含：

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

这些文件用于部署验收和问题排查。GitHub 仓库默认忽略运行目录与日志；完整冻结 release 保留原正式运行证据。

## 常见问题

### UCM 版本警告

当前 UCM 上游显式面向 vLLM 0.9.2，而项目运行时版本字符串为 `0.1.dev10+g05f00f8a8`，因此启动时可能出现版本不在显式支持列表中的提示。

项目最终源码已经完成针对当前 vLLM V1 路径的适配，并通过真实推理、Smoke Test 和正式基准测试。若在新 vLLM 版本上出现 patch apply 失败，则需要重新检查 KVConnector API 和调度器/worker 生命周期，不应继续忽略错误。

### 端口占用

默认端口为 8000。可以使用 Python 检查：

```bash
python -c 'import socket; s=socket.socket(); s.settimeout(.3); r=s.connect_ex(("127.0.0.1",8000)); s.close(); print("occupied" if r==0 else "free")'
```

### 原生扩展加载失败

优先检查：

```text
Python ABI
PyTorch 版本
CUDA Toolkit
GPU 架构
GCC / GLIBCXX
LD_PRELOAD
```

新软件栈下应从冻结源码重新构建，而不是直接复制原 AutoDL 的 Golden Runtime。

### 单请求没有 SSD dump/load

这是正常现象。成本感知 UCM 只有在预计迁移收益大于传输与重新加载成本时才会选择外迁；单请求低压力环境通常没有外迁必要。

## 在不同环境中的复现层级

完整冻结 release 可以区分三种复现目标。

**精确已验证环境复现**使用原 AutoDL 软件栈、冻结源码、Golden Runtime 和原配置。这提供最强的历史运行一致性，但依赖原 ABI。

**源码重建复现**使用两个最终源码仓库，在相近 Linux / NVIDIA / Python / CUDA 环境中重新构建 vLLM 与 UCM 原生扩展，再执行同一配置、Smoke Test 和基准测试。这是 GitHub 公开仓库推荐的复现方式。

**跨版本移植**适用于 Python、PyTorch、CUDA、GPU 架构或 vLLM/UCM 接口变化较大的环境。此时需要重新检查接口兼容、原生扩展、Prefill 曲线和 KVConnector 生命周期，并重新完成 Smoke Test 与正式实验。

## 与基准测试的关系

部署验收只回答系统是否能够正确启动、推理、读取指标并正常停止。V/U/C/F 的性能比较、EnvBench 派生工作负载、聚合方法与结果解释统一记录于：

```text
docs/BENCHMARK.md
```

源码兼容与系统修改见：

```text
docs/COMPATIBILITY.md
```

源码提交、标签与冻结资产关系见：

```text
docs/SOURCE_PROVENANCE.md
```
