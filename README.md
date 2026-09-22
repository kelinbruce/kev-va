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

## 昇腾环境上机验证与验收指南

登录到华为昇腾 910 机器（CANN 9.0.1 运行时环境）后，执行以下步骤完成剩余的 11 项验收任务：

### 0. 克隆仓库与初始化子模块 (前置必须)
本项目依赖三个子模块（`vllm-ascend`、`vllm-upstream`、`kev-reference`）：

- **全新克隆**（带子模块递归）：
  ```bash
  git clone --recurse-submodules https://github.com/kelinbruce/kev-va.git -b gemini-dev
  cd kev-va
  ```
- **如果已经执行过 `git clone`**，在仓库根目录拉取并初始化子模块：
  ```bash
  git submodule update --init --recursive
  ```

### 1. 收集环境与确认基线 (Task 1.1 & 1.2)
```bash
python3 vllm-ascend/tools/kev/collect_environment.py \
    --vllm-root ./vllm-upstream \
    --ascend-root ./vllm-ascend \
    --output ./evidence/env_report.json
```

### 2. 合并并导出 KEV Qwen3 模型制品 (Task 2.5)
```bash
python3 vllm-upstream/tools/kev/export.py \
    --checkpoint <path_to_lora> \
    --base <path_to_qwen3_4b_base> \
    --dtype bfloat16 \
    --output ./kev-qwen3-4b-exported
```

### 3. 启动 SystemOne 服务 (Task 3.4 & 7.1)
```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model ./kev-qwen3-4b-exported \
    --gpu-memory-utilization 0.9 \
    --enforce-eager
```

### 4. 发起 Smoke 验证测试 (Task 7.1)
```bash
curl -X POST http://localhost:8000/v1/systemone \
    -H "Content-Type: application/json" \
    -d '{
      "model": "kev-qwen3-4b",
      "state": "商品尺码错误，且银行卡被重复扣款。",
      "questions": {
        "department": {
          "type": "choice",
          "instructions": "哪个部门优先处理？",
          "criteria": {
            "billing": "重复扣款和支付问题",
            "returns": "退换货和尺码问题"
          }
        },
        "urgency": {
          "type": "score",
          "instructions": "处理紧急程度",
          "criteria": ["低", "中", "高"]
        },
        "duplicate_charge": {
          "type": "noul",
          "instructions": "是否发生重复扣款？"
        }
      }
    }'
```

### 5. 运行数值一致性比对 (Task 7.3 & 7.4)
在 200+ 冻结测试集上比对，要求平均误差 $\le 0.002$：
```bash
python3 vllm-ascend/tools/kev/run_parity.py \
    --inputs vllm-ascend/tests/e2e/kev/data/v1/parity.jsonl \
    --model ./kev-qwen3-4b-exported \
    --mode vllm \
    --output ./evidence/parity_results.jsonl
```

### 6. 性能吞吐与时延扫点 (Task 7.7)
```bash
python3 vllm-ascend/tools/kev/benchmark.py \
    --model ./kev-qwen3-4b-exported \
    --server http://localhost:8000
```

