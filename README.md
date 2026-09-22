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
├── evidence/            # 交付凭据：环境报告、特性支持矩阵、上游与昇腾补丁及 SHA256 校验
│   ├── vllm_upstream_kev.patch
│   ├── vllm_ascend_kev.patch
│   ├── task_1_report.md
│   └── task_8_4_matrix.md
├── vllm-ascend/         # vLLM-Ascend 昇腾适配与测试套件 (分支: codex/kev-qwen3)
├── vllm-upstream/       # 上游 vLLM 核心逻辑实现与测试 (分支: codex/kev-qwen3)
├── kev-reference/       # KEV 原始参考实现引用 (固定 commit: 90990a5f)
└── .agent/              # OpenSpec Agent 技能与工作流配置
```

## 交付与补丁说明

- 上游 vLLM 补丁：[`evidence/vllm_upstream_kev.patch`](evidence/vllm_upstream_kev.patch)
- vLLM-Ascend 补丁：[`evidence/vllm_ascend_kev.patch`](evidence/vllm_ascend_kev.patch)
- 详细规格与任务进展请参考 [`openspec/changes/add-kev-qwen3-decision-serving/`](openspec/changes/add-kev-qwen3-decision-serving/)。
