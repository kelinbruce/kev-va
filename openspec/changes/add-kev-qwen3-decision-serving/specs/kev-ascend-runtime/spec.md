## Purpose

使 Ascend 决策服务的环境、支持配置、运行状态和性能证据可验证，尤其区分用户提供的 CANN 组合、官方配套版本和实际测试结果，防止将启动成功、dummy 输出或文档完成误认为模型适配完成。

## ADDED Requirements

### Requirement: Report actual environment compatibility

环境诊断 SHALL 记录具体 910 型号/显存、驱动与固件、实际容器内 CANN/NNAL、PyTorch/torch-npu、Triton Ascend、镜像 digest 和 vLLM/Ascend commit。0.26 与 CANN 9.0.1 的组合 SHALL 标记为待验证，直至真实基线证据通过；服务工具不得自动升级或降级用户环境。

#### Scenario: User environment differs from official matrix

- **WHEN** 收集到 vLLM-Ascend 0.26 和实际加载的 CANN 9.0.1
- **THEN** 报告与官方 9.1.0 配套要求的差异，并分别记录普通 Qwen3 基线和 KEV 基线结果

#### Scenario: Baseline runtime failure

- **WHEN** 普通 Qwen3 真实权重基线因 CANN/算子/依赖兼容问题失败
- **THEN** 记录复现命令和错误，将环境验收保持未通过，不归因于尚未验证的 KEV 输出头

### Requirement: Enforce the initial supported execution profile

第一版 SHALL 仅接纳单卡 TP=1、明确 dtype、eager、无量化、无运行时动态 LoRA、无 APC 和无 chunked prefill 的配置。BF16 是建议 backbone 精度；硬件不支持时 SHALL 明确报错或要求显式选择并验证 FP16，不能静默改变 dtype。

#### Scenario: Unsupported optimization flag

- **WHEN** 用户为第一版 KEV 实例开启 APC、chunked prefill、图执行、多卡或量化
- **THEN** 启动给出不支持的配置项并拒绝该组合，而非带潜在不完整 readout 继续服务

#### Scenario: Whole-sequence scheduling budget

- **WHEN** 不分块配置下 max_num_batched_tokens 小于允许的最长分支
- **THEN** 启动校验失败，提示调整 token 预算或输入限额

### Requirement: Readiness follows executable model initialization

服务 SHALL 在制品校验、模型初始化和有效决策 warm-up 成功后才报告 ready。模型信息 SHALL 能关联规范模型 ID、制品 revision、dtype 与温度。引擎故障导致模型不可执行时 SHALL 撤销 ready，直至恢复并重新完成 warm-up；可恢复的单请求失败 SHALL 清理该请求而不污染后续请求。

#### Scenario: False-ready prevention

- **WHEN** HTTP 进程已启动但模型 warm-up 失败
- **THEN** ready 检查失败，决策请求不返回 HTTP 200，错误原因可在运行日志中定位

### Requirement: Separate logical usage from executed work

服务 SHALL 分别记录逻辑输入 tokens、实际执行输入 tokens、完成的问题数/请求数、队列时间、端到端时间、错误/取消/超时数和在途请求数。性能报告 SHALL 使用 questions/s、requests/s 和实际输入 tokens/s，不使用序列化 output_tokens 计算解码速度。

#### Scenario: Successful repeated-state batch

- **WHEN** 完整执行 Q 个问题，state 长度 S，问题后缀总长为 L
- **THEN** 逻辑输入为 S+L，实际执行输入为 Q*S+L，神经解码步数为零

#### Scenario: Abort before all branches execute

- **WHEN** 请求取消且部分问题尚未执行
- **THEN** 实际工作计量仅包含已执行部分，取消的剩余工作不计入成功吞吐

### Requirement: Real-weight evidence is mandatory

最终适配报告 SHALL 包含完整真实权重的三类 API 请求、数值一致性、隔离性、边界输入、并发、失败清理和性能基线结果，标明输入集和制品校验值。dummy smoke SHALL 仅证明执行路径，不得作为模型效果或最终验收证据。

#### Scenario: Dummy-only success

- **WHEN** 只有 dummy 权重的启动和 HTTP 200 证据
- **THEN** 报告将真实权重验收标记为未完成，不能将能力标成已验证支持

#### Scenario: Completed baseline report

- **WHEN** 真实权重通过全部功能与数值门槛
- **THEN** 发布可复现命令、版本、原始指标和已知限制；性能基线如实报告，不声称达到尚未约定的业务延迟目标
