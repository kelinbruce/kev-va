## Context

动机和首版范围见 [proposal.md](proposal.md)。本设计处于提案状态，尚未实现或在 NPU 上验证。

已核对的来源：

| 项目 | 固定来源 |
| --- | --- |
| KEV 参考代码 | `jaredpalmer/kev@90990a5fac2995b9faa3190f7d437e84f2067768` |
| 建议模型 | `jaredpalmer/kev-4b`，revision `qwen3` |
| 模型 revision 当前解析值 | `c4bfa11b0dc07691884f2d97f1c4c4c05c92e416` |
| 基座 | `Qwen/Qwen3-4B-Base`；基座 SHA 需从制品 provenance/head 元数据解析并验证 |
| 已核对 adapter 配置 | rank=16，alpha=32，覆盖 Q/K/V/O 和 gate/up/down 投影；没有 trainable token indices |
| 上游 vLLM 参考 | `v0.26.0@568afb3a13806beb53bb2e6bd518269357b237c0` |
| Ascend 参考 | `v0.26.0rc2@155f68974c802633c19650ece430d41ee6802fa6` |
| 当前交付仓库起点 | `releases/v0.26.0rc@9aebd4dd8f3379f0d171bbdc4ec87544cd6aeb3c` |

以上是本提案调研快照；实施时记录实际检出的源代码，不将不同 commit 的测试结果混用。

## Goals / Non-Goals

**Goals:**

- 从同一份固定制品生成可追溯的决策概率，保持 KEV 输入和 pointer readout 语义。
- 通过既有 pooling 执行路径接入 Ascend；用序列边界保证问题隔离。
- 使每项外部需求都能由单元测试、引擎集成测试或真实权重报告验证。

**Non-Goals:**

- 第一版不共享 state 计算，不引入 packed tree mask 或自定义 Attention kernel。
- 不重训或重新设计输出头，不声称达到 Jev 的模型能力或校准质量。
- 不实现 Qwen3.5、图像输入、生成/流式接口、MTP、PD 分离、多卡、量化和运行时动态 LoRA。
- 不将通用模型适配技能中的 128k 容量测试作为本模型首版验收范围；按下文显式输入限额验收。

## Decisions

### D1. 先固定环境证据，再验证 KEV

用户环境为 Ascend 910 系列、CANN 9.0.1；官方 0.26 兼容表要求 CANN 9.1.0。此组合属于“待验证”，不能直接写成支持或不支持。

首先收集实际设备型号/显存、驱动、固件、容器 digest、CANN、NNAL、PyTorch、torch-npu、Triton Ascend 和两个代码仓库的版本。应读取容器实际加载的 CANN，而不是仅根据宿主机目录判断。

在原环境执行普通 Qwen3-4B 的真实权重 prefill/推理基线，再执行本决策模型基线。环境不通过时输出可复现错误；代码开发和 CPU 单元测试可继续，NPU 验收保持未通过。升级 CANN 或更换 vLLM-Ascend 版本需另行确定，不能由本提案自动执行。

### D2. 固定模型并离线导出完整制品

首个验收模型采用上表 Qwen3 KEV-4B。导出过程将 LoRA 在 FP32 中合并，再转换 backbone 到明确指定的运行 dtype；pointer head 的两组线性层参数（包括 bias）保留 FP32。读取 head 实际维度和温度，不能用默认值覆盖已存在字段。

导出产物建议采用 Hugging Face 配置、tokenizer 文件和 safetensors，附 `kev_manifest.json`。manifest 至少记录：

- 格式版本、模型 ID、模型/基座 repo 与不可变 SHA、KEV 参考代码 SHA。
- 原 adapter/head 文件 SHA256、导出权重和 tokenizer 文件 SHA256。
- 架构、hidden size、head dimension、backbone/head dtype、LoRA 合并方式、校准温度。
- 五个结构 token 的字符串和 ID、`option_isolation`、支持的协议版本。

若基座训练 revision 未记录，必须在报告中披露并固定所使用基座 SHA，通过参考一致性后才能验收，不能宣称已恢复未知的训练来源。发现缺失权重、维度不符、非有限参数、温度不大于零、结构 token 无效或 `option_isolation=true` 时，首版导出/加载失败并给出原因。

导出工具使用隔离环境，不升级 Ascend 运行容器中的 Transformers。运行时无需加载 PEFT，也不执行来自模型制品的任意远程建模代码。

备选的动态 LoRA 会扩大版本和加载映射验证范围，因此留待后续需求。

### D3. 一条问题对应一条完整因果序列

共享 state 编码为 S；每个问题后缀为 B_i。引擎输入为独立的 `S + B_i`，每条序列 position IDs 从 0 开始，问题位置接续自身 state 长度。

```text
HTTP request
  -> schema / limits / renderer
  -> [state + question A] --\
  -> [state + question B] ----> engine batching -> causal Qwen3
  -> [state + question C] --/                        |
                                            selected hidden states
                                                   |
                                            pointer + temperature
                                                   |
                                            per-question softmax
                                                   |
                                         aggregate by question ID
```

所有选项都在同一问题序列内，保留因果顺序和相互影响。不能拆成“每个选项一条序列”。外部 question ID 只参与映射，不送入模型；choice 的选项名称属于 KEV 原有输入文本，必须保留。

复用固定 KEV 的 `render`、特殊 token 转义和 option_text 规则；直接提交 token IDs，避免 chat template、BOS/EOS 或二次 tokenization 改变输入。用户文本中的伪造结构标记必须先转义。

这种布局在默认非 option-isolation 的纯 Attention 模型上保留参考计算依赖，但重复处理 state。选项换序可能改变模型答案；测试需测量这种偏差，不能把“选项换序结果完全一致”设成迁移正确性要求。

### D4. 用自定义 Pooler 完成 pointer readout

在上游 vLLM 中定义 Qwen3 决策模型，复用 causal Qwen3 backbone 并注册为 pooling 模型。不能自动转换成双向 attention 或套用只读最后位置的分类头。

使用 `EngineClient.encode` / `PoolingParams(task="plugin")`；自定义路由负责预处理，不经过通用 embedding/classification HTTP 序列化。通过 `extra_kwargs` 传入每条序列的选项末尾索引、决策索引和元数据版本；这些索引均为序列内逻辑位置，不预先混入 batch 偏移。启动 profiling 的 dummy 输入须有专门有效元数据路径。

Pooler 使用运行时序列偏移 gather 隐藏状态，计算：

```text
u = W_q h_decide + b_q
v_j = W_k h_option_end_j + b_k
z_j = dot(v_j, u) / sqrt(head_dim)
p = softmax(z / temperature)
```

gather 后的转换、线性投影和 softmax 在 NPU 上进行，head 使用 FP32。每条序列输出一个长度 K_i 的概率张量，返回结果按请求保留变长边界；不对不同问题的候选项联合 softmax，也不额外应用通用 pooling 的归一化或 softmax。

只将概率及必要统计回传 CPU，不回传完整 hidden states，不在候选项循环中执行设备 `item()`。语义验证可通过内部诊断路径获取 logits；对外 API 不暴露 hidden states。

最先完成的引擎集成验证必须覆盖：metadata 跨进程传递、ragged 输出、batch 重排、dummy profiling 和取消。若现有扩展点存在限制，应最小化补充上游通用路径；NPU runner 修改必须有后端必要性证据。

### D5. 专用路由、父子请求生命周期和计量

通用协议和处理器放在上游 vLLM；在决策模型实例中注册 `/v1/systemone`。保留已有认证、模型查询、健康检查和 metrics 设施。生成路由对该实例不可用，不影响其他模型实例。

服务器生成内部父请求 ID 和子请求 ID，不用用户 question ID 作为全局引擎 ID。所有问题完成才返回 HTTP 200；一个问题失败时，取消其他未完成子请求并返回整个请求的错误。

API 层维护有界父请求队列和子请求并发池。就绪的父请求轮流获得子请求提交机会，每个父请求有 fan-out 上限；引擎依旧按 token 预算调度。timeout 包含排队时间。断连、超时和异常都调用引擎 abort，并清理未提交子请求、映射和占用配额。

若 S 为 state（含标记）的 token 数，L_i 为问题后缀 token 数：

- API `usage.input_tokens = S + sum(L_i)`，保持 KEV 的逻辑计量。
- 完整成功请求的实际输入工作量为 `sum(S + L_i)`，单独记录。
- 取消请求只累计实际被执行的 token，不能直接累加预估请求总量。
- `usage.output_tokens` 沿用固定 KEV 的 answers 序列化计数规则，是序列化长度，神经解码步数为零。
- `latency_ms` 记录服务端接收至完成聚合的时间，包含排队；模型时间、排队时间另记 metrics。

### D6. 协议兼容范围及建议默认限额

本提案兼容固定 KEV 的 SystemOne 主体输入/输出语义，不声称完全复刻 TypeSafe 私有实现。具体字段要求见 [API spec](specs/systemone-decision-api/spec.md)。

有意明确的边界：

- `score` 接受 2..255 个等级，与固定 KEV 一致；TypeSafe 当时文档的上限为 10。
- 概率按 KEV 保留两位小数，舍入后的分布不保证精确求和为 1；内部原始概率必须归一化。
- choice 的胜出选项、score 的期望和 confidence 均先用原始概率计算，再舍入；并列时按输入顺序。
- 不静默截断 state；超长返回 422。这是相对 KEV 默认截断行为的有意收紧。
- 未知字段、生成参数、无法识别的 model 和非有限数值不静默忽略。

以下为提案建议默认值，可由显式服务配置调整；并非已经测得的容量或性能承诺：

| 配置 | 建议值 | 约束 |
| --- | --- | --- |
| 单请求问题数 | 1..32 | 校验后才提交子请求 |
| choice 候选项 | 1..255 | 顺序保留 |
| score 等级 | 2..255 | 值为 0..K-1 的等级索引 |
| state tokens | 最大 8192，含标记 | 还须满足分支总长度 |
| 单分支总 tokens | 最大 8192 | state + 一个完整问题 |
| 单请求逻辑 tokens | 最大 32768 | state 只计一次 |
| 单实例已接纳父请求 | 最大 32 | 包含排队和执行中 |
| 单父请求在途子请求 | 最大 4 | 防止一个父请求一次占满所有配额 |
| 全局在途子请求 | 最大 16 | 配合 engine max_num_seqs |
| 请求 deadline | 30 秒 | 包含排队和推理 |
| engine max_num_seqs | 16 | 不代表 HTTP 请求数 |
| engine max_num_batched_tokens | 8192 起 | 至少等于允许的最长完整分支 |

容量型配置和 deadline 在启动时检查合法性并记录。HTTP body 上限沿用部署网关/服务器的独立限制；token 限额不能代替字节限额。

示例请求：

```json
{
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
    "duplicate_charge": {
      "type": "noul",
      "instructions": "是否发生重复扣款？"
    },
    "urgency": {
      "type": "score",
      "instructions": "处理紧急程度",
      "criteria": ["低", "中", "高"]
    }
  }
}
```

响应字段示例（概率仅用于展示格式，不代表实测结果）：

```json
{
  "model": "kev-qwen3-4b",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "confidence": 0.46,
      "probabilities": {"billing": 0.73, "returns": 0.27}
    },
    "duplicate_charge": {"type": "noul", "noul": 0.95},
    "urgency": {
      "type": "score",
      "score": 1.1,
      "legend": {"0": "低", "1": "中", "2": "高"},
      "probabilities": {"0": 0.1, "1": 0.7, "2": 0.2},
      "confidence": 0.85
    }
  },
  "usage": {"input_tokens": 100, "output_tokens": 120},
  "latency_ms": 25.0
}
```

### D7. 明确上游与 Ascend 的交付位置

以下为拟定落点，实际文件名可在实现中按上游组织微调，组件职责不变：

| 组件 | 目标位置 |
| --- | --- |
| Qwen3 决策模型与权重加载 | 上游 `vllm/model_executor/models/kev_qwen3.py` 及 registry |
| pointer Pooler | 上游 `vllm/model_executor/layers/pooler/` |
| 协议、renderer、服务与聚合 | 上游 `vllm/entrypoints/pooling/systemone/` |
| 路由与初始化 | 上游 API server 按模型能力注册 |
| 制品导出工具 | 上游 `tools/` 内独立工具 |
| Ascend 特定问题 | 当前仓库 `vllm_ascend/patch/worker/` 或必要的后端组件 |
| 本仓库测试与说明 | `tests/ut/`、`tests/e2e/`、`docs/source/tutorials/models/` |

当前仓库只有 Ascend 源码。实施交付采用两个对应的源代码 diff：上游 vLLM 独立 commit/diff，以及本仓库的后端适配和验收材料。在本仓库交付记录中引用准确上游 SHA，并附可应用于固定 vLLM 基线的 patch 与 SHA256，避免依赖临时目录的隐含修改。

不直接在 Ascend 仓库新增通用模型文件，不全局修改所有 Qwen3 的 head 或 attention 行为。

### D8. 分层验收与数值阈值

所有指标使用未舍入概率。平均概率绝对误差先在每个问题的 K 个选项内求均值，再对问题求均值；最大误差取所有问题/选项的最大值。argmax 一致率按问题等权计算。以下阈值是本提案的初始验收标准；若实测未通过，应定位差异并明确修订，不能在脚本中静默放宽。

| 验证层 | 基线和通过条件 |
| --- | --- |
| 输入与后处理 | tokenizer IDs、position IDs、readout 索引、键顺序和确定性后处理与固定参考完全一致 |
| pointer 单元测试 | 同一组 FP32 hidden states 的 logits/probabilities，atol=1e-5、rtol=1e-5 |
| 导出一致性 | 同一参考环境，合并前后 FP32 全模型最大概率绝对误差 <=1e-4；记录库和硬件 |
| Ascend 迁移一致性 | 固定真实权重、同一 backbone dtype 的 KEV 参考与 NPU：平均概率绝对误差 <=0.002，最大 <=0.02；argmax 一致率 >=99% |
| 明确胜出项 | 参考 top1-top2 概率差 >0.04 的问题，argmax 必须全部一致；K=1 单独验证 |
| 问题隔离/批处理 | 同一 NPU 配置下单独、批量、重排问题、混入其他请求，最大概率差 <=0.005；margin >0.01 的 argmax 不变 |
| 概率合法性 | 所有原始概率有限且在 [0,1]，每题求和误差 <=1e-6；每个选项恰好有一个结果 |

冻结不少于 200 个问题的迁移一致性集，三种类型各不少于 30 个，包含中文、英文、嵌套 JSON、特殊标记字符串、不同长度和候选项数。单选项、255 候选项、最大输入长度与超限输入作为独立边界集。保存输入、模型版本、原始参考结果和数据 SHA256。

FP32 参考导出检查、同 dtype NPU 迁移检查和 BF16 相对 FP32 的量化误差分别报告。若实际 910 不支持建议的 BF16 路径，显式选择 FP16 并重新生成同 dtype 比较，禁止静默转换。

报告阈值 0.5、0.8、0.9 附近的业务动作变化、选项排列敏感性以及带标签样本的准确率/Brier/ECE；不承诺迁移能改善原模型校准。参考概率距离业务阈值超过 0.02 的样本不得跨越该阈值。

功能通过后做性能基线，至少覆盖 state 长度 128/1024/4096、问题数 1/4/16、候选项数 2/8/32、HTTP 并发 1/4/16 的单因素扫点及一个多因素组合。只执行满足 token 限额的组合。记录 warm-up、样本数、P50/P95、questions/s、requests/s、实际输入 tokens/s、峰值显存、失败和超时数。相同硬件/精度的 KEV 参考能运行时附对照；第一版不设未经业务确认的延迟或加速倍数门槛。

APC 始终关闭，因此区分服务冷启动/算子预热，不将结果标成“前缀缓存命中加速”。

## Risks / Trade-offs

- [CANN 与官方矩阵不同] → 以真实环境诊断和基线验证为门槛，分别记录环境故障与 KEV 故障。
- [独立问题重复处理 state] → 明确实际 token 开销；以有界 fan-out 控制资源，后续单独设计共享前缀。
- [prefix cache 跳过 readout 位置] → 第一版禁止 APC；后续需限制命中边界或增加 readout 表示缓存。
- [chunked prefill 丢失前段选项表示] → 第一版禁止 chunked prefill；后续按请求跨 chunk 保存选项投影。
- [原参考训练上下文较短] → 首版最大长度是工程容量，不等于长上下文质量保证；单独报告长输入效果。
- [Pooler profiling、跨进程 metadata 或变长输出不符合假设] → 在正式 API 接入前完成引擎路径验证。
- [精度和选项顺序影响阈值动作] → 分层数值验证，保留原始分布并报告动作翻转。

## Migration Plan

1. 固定源代码与模型版本，完成环境基线和参考结果集；不修改现有服务实例。
2. 在匹配上游 vLLM 中完成通用能力，在 Ascend 上验证必要后端适配。
3. 先执行 dummy 路径 smoke，再执行完整层数的真实权重一致性和 API 场景；dummy 不作为质量证据。
4. 启动独立 KEV 决策实例，发布准确镜像/模型 digest、运行命令和验收报告。
5. 回滚时停止该实例并恢复先前镜像、代码和模型制品；本方案不要求修改其他 Qwen3 实例。
6. 共享前缀、分块或图执行在后续 OpenSpec 变更中重新定义配置组合与验收，不能悄悄打开。

## References

- [KEV 模型实现](https://github.com/jaredpalmer/kev/blob/90990a5fac2995b9faa3190f7d437e84f2067768/kev/model.py)
- [KEV API 与后处理](https://github.com/jaredpalmer/kev/blob/90990a5fac2995b9faa3190f7d437e84f2067768/kev/api.py)
- [KEV checkpoint 加载](https://github.com/jaredpalmer/kev/blob/90990a5fac2995b9faa3190f7d437e84f2067768/kev/checkpoint.py)
- [Qwen3 模型 revision](https://huggingface.co/jaredpalmer/kev-4b/tree/c4bfa11b0dc07691884f2d97f1c4c4c05c92e416)
- [vLLM 0.26 PoolingParams](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/pooling_params.py)
- [vLLM 0.26 PoolingMetadata](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/pool/metadata.py)
- [Ascend 0.26 安装环境](https://docs.vllm.ai/projects/ascend/en/v0.26.0rc1/installation.html)
- [TypeSafe API](https://docs.typesafe.ai/api)
