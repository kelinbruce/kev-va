## Purpose

保证 Ascend 上的决策概率保留固定 KEV Qwen3 参考实现的输入、信息可见性与数值含义，使多问题和跨请求批处理不会改变问题边界，并为真实权重迁移提供明确的验收条件。

## ADDED Requirements

### Requirement: Reference-compatible input representation

服务 SHALL 保持固定 KEV renderer、tokenizer、结构标记、选项名称/描述、选项顺序及位置的语义；用户 question ID SHALL 仅用于响应映射。用户文本中的结构标记 SHALL 转义，服务不得额外插入 chat template、BOS 或 EOS。

#### Scenario: Structured input with control-like text

- **WHEN** state 和 instructions 含嵌套 JSON、中文、布尔值及形如 `<|fim_suffix|>` 的用户文本
- **THEN** token IDs 和选项/决策位置与固定参考一致，用户文本不能创建额外结构边界

#### Scenario: Rename a question

- **WHEN** 只改变 question ID 而不改变 state、instructions 和 criteria
- **THEN** 仅响应键发生变化，模型输入和概率结果保持一致

### Requirement: Isolate questions and preserve option interaction

每个问题 SHALL 仅使用共享 state 和该问题自身的指令及全部选项。问题之间、请求之间 SHALL 没有信息可见性；同一问题的选项 SHALL 保留原有因果顺序，不能作为相互独立的问题评分。

#### Scenario: Sibling question contains unrelated information

- **WHEN** 在兄弟问题中加入另一问题不应看到的秘密字符串，或增删、重排兄弟问题
- **THEN** 被测问题的可见输入不变，结果满足批处理一致性阈值

#### Scenario: Mixed requests in one batch

- **WHEN** 两个不同 state 的请求共享一个推理 batch
- **THEN** 每个问题的概率只对应自身请求的 state 和 options，不能发生结果错配

#### Scenario: Option ordering

- **WHEN** 同一个问题改变选项顺序
- **THEN** 模型按新顺序重新计算并正确映射候选项；系统不承诺不同顺序的概率相同

### Requirement: Direct calibrated decision distribution

系统 SHALL 从决策位置和所有选项末尾的最终表示计算 checkpoint 定义的 pointer 分数，经相同温度校准后逐问题归一化。系统 SHALL 不执行自回归文本生成。每个问题 SHALL 恰好返回 K 个有限原始概率，值位于 [0,1]，求和误差不超过 1e-6。

#### Scenario: Different numbers of options

- **WHEN** 同一 batch 中的问题分别有 1、2、8 和 255 个选项
- **THEN** 各自得到对应长度的概率分布，单选项概率为 1，其他问题不参与该题归一化

#### Scenario: Readout uses multiple positions

- **WHEN** 决策位置相同但不同选项末尾表示导致参考 pointer 分数不同
- **THEN** 结果反映全部对应位置的计算，而不是只使用最后一个 token 的固定类别分类结果

### Requirement: Real-weight numerical parity

迁移验收 SHALL 使用固定真实权重、同 backbone dtype 的参考结果和未舍入概率。冻结集 SHALL 不少于 200 个问题，三种类型各不少于 30 个；平均概率绝对误差 SHALL 先对题内选项求均值再对问题求均值，并且不超过 0.002；所有问题/选项的最大误差 SHALL 不超过 0.02，按问题等权的 argmax 一致率 SHALL 至少 99%。参考 top1-top2 差大于 0.04 的问题 SHALL 全部保持 argmax。

#### Scenario: Compare the NPU model with the reference

- **WHEN** 在完整层数、真实权重和固定数据集上执行迁移比较
- **THEN** 报告上述所有指标、失败样本、精度和版本；任一阈值未通过时不能宣布数值验收完成

#### Scenario: Business threshold decisions

- **WHEN** 比较概率在 0.5、0.8、0.9 阈值两侧的动作
- **THEN** 报告动作翻转，参考概率距离对应阈值大于 0.02 的样本不得跨越阈值

### Requirement: Batch-invariant question results within tolerance

在相同已验证 NPU 配置下，单独执行、混合 batch、变长 padding 和问题重排之间，逐题最大概率绝对差 SHALL 不超过 0.005；原始 top1-top2 差大于 0.01 的问题 SHALL 保持 argmax。

#### Scenario: Rebatch an unchanged question

- **WHEN** 同一问题与不同长度、不同候选项数的其他问题重新组 batch
- **THEN** 结果满足阈值且选项/问题映射不变，出现超差时验收失败并保留样本
