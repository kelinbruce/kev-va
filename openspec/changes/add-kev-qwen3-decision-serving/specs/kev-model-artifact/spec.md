## Purpose

提供能够离线加载、验证来源并重复产生决策结果的 Qwen3 KEV 模型制品，使开发、测试和部署使用同一组基座、训练参数、tokenizer 与校准配置，避免浮动版本造成结果不可追溯。

## ADDED Requirements

### Requirement: Immutable model provenance

模型制品 SHALL 记录基座与 KEV 权重的 repo、不可变 revision、参考代码 revision、原始 adapter/head 校验值、导出文件校验值及格式版本。首个验收配置 SHALL 使用 Qwen3 KEV-4B 的固定 qwen3 revision；不得把浮动 main 当作已固定的 Qwen3 权重。

#### Scenario: Resolve the Qwen3 checkpoint

- **WHEN** 从 `jaredpalmer/kev-4b` 的 `qwen3` revision 导出
- **THEN** 制品记录其不可变 SHA 和 `Qwen/Qwen3-4B-Base` 基座身份，并能在无网络环境中加载全部必需文件

#### Scenario: Missing base training revision

- **WHEN** 上游制品没有记录基座训练 SHA
- **THEN** 导出报告明确标记来源缺口，固定实际使用基座 SHA，并要求参考一致性通过后才允许完成验收

### Requirement: Preserve trained decision parameters

制品 SHALL 保留训练后的 backbone 更新、全部 pointer 参数、结构 token 映射、head dimension 和 checkpoint 校准温度；离线导出 SHALL 在记录的环境中验证 FP32 合并前后最大概率绝对误差不超过 1e-4。运行精度转换 SHALL 单独记录，不得与合并误差混为一项。

#### Scenario: Export a trained checkpoint

- **WHEN** 导出带 LoRA、pointer 权重和有效温度的 Qwen3 checkpoint
- **THEN** 加载后的模型使用训练后的参数和相同温度，报告合并误差，并保留 head 的 FP32 精度

#### Scenario: Non-default calibration

- **WHEN** checkpoint 携带非 1.0 的正温度
- **THEN** 推理使用该温度，模型信息可查询该值，且不得静默替换成默认温度

### Requirement: Reject incompatible or corrupt artifacts

导出与加载 SHALL 验证文件校验值、参数形状、有限数值、温度、tokenizer 和架构一致性。第一版 SHALL 拒绝 Qwen3.5、option-isolation 模式、缺失训练参数及无效结构 token，并返回可定位原因，不能回退到随机 head 或未适配基座。

#### Scenario: Corrupt or incomplete weights

- **WHEN** 文件校验失败，或 head 的一个 bias/weight 缺失
- **THEN** 加载失败，服务不进入可接收决策请求的就绪状态

#### Scenario: Wrong architecture

- **WHEN** 将最新 main 的 Qwen3.5 制品传给第一版服务
- **THEN** 明确报告架构不在支持范围，且不尝试使用 Qwen3 的加载映射继续启动

### Requirement: Reproducible export without runtime dependency upgrades

导出流程 SHALL 记录工具版本和参数，并与 Ascend 运行依赖隔离。运行制品 SHALL 不依赖加载时下载 adapter、不依赖远程建模代码，也不得为导出自动升级服务容器的 Transformers。

#### Scenario: Deploy the exported model offline

- **WHEN** 在固定 Ascend 镜像中加载已完整导出的制品且网络不可用
- **THEN** 服务能够完成制品校验和加载，或仅报告真实缺失/不兼容项，而不是尝试在线补齐权重
