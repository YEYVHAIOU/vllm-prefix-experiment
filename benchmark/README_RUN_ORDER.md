# 基准测试运行顺序

本文件提供 V/U/C/F 实验的最短运行入口。完整工作负载定义、指标计算和结果解释见 [`docs/BENCHMARK.md`](../docs/BENCHMARK.md)。

## 四种配置

| 配置 | 调度器 | Continuum | UCM |
|---|---|---:|---:|
| V | FCFS | 关闭 | 关闭 |
| U | FCFS | 关闭 | eager/full |
| C | Continuum | 开启 | 关闭 |
| F | Continuum | 开启 | `cost_full` + Frontier-Tail + TieredStore |

四组共享同一最终 vLLM/Continuum 代码基座；U/F 使用同一最终 UCM 代码版本。

## 运行前准备

先确认三个仓库位于同级目录，并已经切换到冻结源码版本：

```text
workspace/
├── vllm-prefix-experiment/
├── vllm-continuum/
└── unified-cache-management-continuum/
```

设置本机 Python 环境和模型目录：

```bash
export VENV_PATH=/path/to/your/vllm-environment
export MODEL_PATH=/path/to/Qwen3-0.6B
```

如源码不在默认同级目录，可额外设置：

```bash
export VLLM_REPO=/path/to/vllm-continuum
export UCM_REPO=/path/to/unified-cache-management-continuum
```

## 推荐运行顺序

先执行四配置 Smoke Test：

```bash
bash benchmark/scripts/smoke_four_cases.sh
```

随后可运行 C76 小规模验证：

```bash
bash benchmark/scripts/run_c76_pilot.sh
```

压力与并发承载实验使用：

```bash
bash benchmark/scripts/run_vucf_capacity.sh
```

冻结的压力实验由 `run_c76_pilot.sh` 完成 C76，并由 `run_vucf_capacity.sh` 完成 C96 / C128。若要扩展到更高并发，需要先构造对应 trajectory 工作负载，再显式设置新的并发档位。

正式 Full Runtime-Eligible EnvBench 派生实验使用：

```bash
bash benchmark/scripts/run_full_envbench_vucf.sh
```

正式工作负载由 9 个确定性分片组成，每个分片依次运行 V → U → C → F。

## 稳定性判据

一次运行至少需要满足：

```text
expected requests == successful requests
failed requests == 0
```

同时服务端不应出现 OOM、EngineCore crash、请求超时或致命运行时错误。

并发 128 是本项目正式覆盖的最高实验点；更高并发需要单独构造并验证，不由现有结果推断。

## 输出位置

压力实验和正式实验结果分别写入 `benchmark/results/` 下对应目录。正式聚合结果固定保存在：

```text
benchmark/results/full_envbench_final_20260823/
├── aggregate_summary.json
├── aggregate_summary.tsv
├── aggregate_comparison.md
└── manifest.tsv
```

请求级运行目录、服务日志和 SSD backing 默认属于运行时产物，不进入普通 Git 历史。

## 结果聚合

正式 9 分片实验完成后，聚合逻辑由：

```text
benchmark/scripts/aggregate_full_envbench.py
```

负责。脚本重新汇总请求级 TTFT / TPOT / E2E、trajectory 级 JCT、Prefix Cache、Dynamic TTL、KV Cache 周期采样和 UCM/SSD 数据，并生成 JSON、TSV 与 Markdown 三种聚合输出。
