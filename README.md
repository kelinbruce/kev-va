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
├── vllm-ascend/         # vLLM-Ascend 基线及后续昇腾适配
├── vllm-upstream/       # 上游 vLLM v0.26.0 基线及后续通用实现
├── kev-reference/       # KEV 原始参考实现引用 (固定 commit: 90990a5f)
└── .agent/              # OpenSpec Agent 技能与工作流配置
```

## 当前实施状态

`gpt-dev-fresh` 从未实施状态开始。OpenSpec 的 proposal、design 和 specs 保留为开发输入，任务清单已全部重置；代码、测试与验收证据将按任务完成情况重新产生。

详细规格与任务进展请参考 [`openspec/changes/add-kev-qwen3-decision-serving/`](openspec/changes/add-kev-qwen3-decision-serving/)。
