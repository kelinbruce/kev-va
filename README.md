# kev-va: KEV Decision Serving on vLLM-Ascend

本项目为在华为昇腾（Ascend 910 系列）芯片上基于 vLLM-Ascend 与上游 vLLM 接入 KEV (Qwen3) 专用决策推理服务的研发与交付工作区。

## 项目概述

- **目标**：在 Ascend NPU 环境下，为基于 KEV 结构的 Qwen3 决策模型提供 SystemOne 风格的单步前向概率推理接口（`/v1/systemone`），支持 `noul`、`choice` 和 `score` 三类问题决策。
- **架构解耦**：
  - 上游 vLLM：承载专用的 SystemOne API、`KevPooler`、`KevQwen3ForCausalLM` 骨干网络及输出头。
  - vLLM-Ascend：承载昇腾环境适配、NPU 运行时配置、E2E/UT 测试套件与部署教程。
- **规范与交付**：采用 OpenSpec 驱动开发规范，全流程记录设计决议、测试基线与验收证据。

## 目录结构

```text
.
├── openspec/            # OpenSpec 规范定义、变更提案与任务跟踪 (add-kev-qwen3-decision-serving)
├── evidence/            # 重新实施过程中产生的环境、测试、补丁及校验证据
├── tools/               # KEV 模型制品离线导出与校验工具
├── tests/               # 仓库级工具测试
├── vllm-ascend/         # vLLM-Ascend 基线及后续昇腾适配
├── vllm-upstream/       # 上游 vLLM v0.26.0 基线及后续通用实现
├── kev-reference/       # KEV 原始参考实现引用 (固定 commit: 90990a5f)
└── .agent/              # OpenSpec Agent 技能与工作流配置
```

## 当前实施状态

`gpt-dev-fresh` 从未实施状态开始。OpenSpec 的 proposal、design 和 specs 保留为开发输入，任务清单已全部重置；代码、测试与验收证据将按任务完成情况重新产生。

详细规格与任务进展请参考 [`openspec/changes/add-kev-qwen3-decision-serving/`](openspec/changes/add-kev-qwen3-decision-serving/)。

## 模型制品工具

工具不会下载模型，也不会修改 Ascend 运行环境。准备好固定 revision 的 checkpoint
目录和 Qwen3 基座目录后，在独立 Python 环境中运行：

```bash
python tools/kev_artifact.py export \
  --source /path/to/kev-4b-checkpoint \
  --base /path/to/Qwen3-4B-Base \
  --provenance evidence/sources/qwen3-checkpoint.json \
  --output /path/to/kev-qwen3-4b-artifact \
  --dtype bfloat16

python tools/kev_artifact.py validate /path/to/kev-qwen3-4b-artifact
```

导出会在 FP32 中合并 LoRA，按指定 dtype 写出 backbone，保留 FP32 pointer
head，并生成带来源、依赖版本、参数和全部文件 SHA256 的 `kev_manifest.json`。
目标目录已存在时默认拒绝覆盖；只有显式传入 `--force` 才会替换目标目录。

## 真实权重验收与 benchmark 入口

[`configs/kev_real_weight_acceptance.json`](configs/kev_real_weight_acceptance.json)
固定真实权重验收数据路径、数值阈值、边界/生命周期覆盖项和性能矩阵。先在 Ascend
环境准备由固定 tokenizer 核验过 token 数的 state workload；工具本身不会下载模型：

```bash
python tools/kev_benchmark.py plan

python tools/kev_benchmark.py run \
  --case s128-q1-k2-c1 \
  --states /data/kev/verified-state-workloads.json \
  --output /results/kev-s128-q1-k2-c1.json
```

`plan` 包含 state tokens 128/1024/4096、问题数 1/4/16、候选项数 2/8/32、并发
1/4/16 的单因素扫点和一个多因素组合。`run` 记录 warm-up、样本数、P50/P95、
requests/s、questions/s、逻辑 tokens/s、从 `/metrics` 读取的实际 tokens/s 和失败
详情。峰值 NPU 显存仍须从 Ascend 运行时采集并与结果一起归档；工具明确将其标记为
缺失，不会伪造数值。
