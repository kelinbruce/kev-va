# 任务 8.4：交付状态矩阵

状态口径：`通过` 仅表示表中指定层级已有可复现证据；`待验证` 不表示失败或支持。
本次开发未下载模型，因而不产生真实权重结论。

## 环境与执行层级

| 层级 | 环境/制品 | 状态 | 证据或缺口 |
|---|---|---|---|
| 本地开发 | macOS、CPU、Python 3.12 | 通过 | artifact、renderer、pointer、API、调度、readiness 与 metrics 单测 |
| 上游引擎 dummy | CPU dummy pooling、进程边界序列化 | 通过 | K=2/K=3 ragged 输出及 request ID 对应测试 |
| Ascend 环境诊断 | Ascend 910、驱动/固件、CANN/NNAL、torch-npu、镜像 | 待验证 | 当前主机无 NPU；未采集实际容器版本和 import path |
| Ascend dummy | vLLM 0.26 + vLLM-Ascend 0.26 | 待验证 | 必须在实际设备运行三类型 smoke |
| 模型制品 | 固定来源元数据和导出/校验工具 | 部分通过 | 工具和拒绝场景通过；未读取/导出真实权重 |
| Ascend 真实权重 | KEV Qwen3-4B 完整制品 | 待验证 | 未下载模型；未执行离线加载、API、精度、并发或性能测试 |

CANN 9.0.1 与官方 0.26 配套目标 CANN 9.1.0 不同，所以 9.0.1 组合保持
`待验证`。没有用升级环境或替换版本来掩盖该差异。

## 数值与功能

| 项目 | 状态 | 已验证范围 | 尚需真实环境验证 |
|---|---|---|---|
| KEV renderer/token 索引 | 通过 | 嵌套 JSON、中文、布尔/null、伪结构标记、位置/readout | 真实 tokenizer 全集复核 |
| FP32 pointer | 通过 | bias、温度、K=1/2/8/255、ragged batch，atol/rtol 1e-5 | NPU kernel 与同 dtype 参考误差 |
| 三类型 API/后处理 | 通过 | noul/choice/score、并列、两位小数、未知字段/模型 | 真实权重响应概率 |
| 请求限额与原子性 | 通过 | 全请求预检、无静默截断、乱序聚合、分支失败 | NPU 高并发恢复 |
| deadline/断连/abort | CPU 通过 | abort、资源回落、错误码、成功吞吐计量 | NPU 执行中取消与后续恢复 |
| 制品安全门 | 通过 | manifest、hash、结构 token、配置、head、warm-up gate | 完整真实制品离线加载 |
| 迁移精度 | 待验证 | 仅 synthetic FP32 单元数值 | 平均 <=0.002、最大 <=0.02、argmax >=99% |
| 批处理不变性 | 部分通过 | synthetic batch 重排/混排与映射 | 真实权重最大差 <=0.005 |
| 校准/业务阈值 | 待验证 | 后处理公式和稳定并列规则 | 0.5/0.8/0.9、Brier/ECE、长上下文/中文 |
| 性能 | 待验证 | 无真实性能声明 | P50/P95、questions/s、requests/s、tokens/s、显存 |

## 首版范围与后续范围

| 能力 | 首版状态 | 说明 |
|---|---|---|
| 单卡 TP=1、显式 dtype、eager | 唯一允许配置 | 启动校验强制执行 |
| 共享前缀/APC | 后续范围 | 首版禁用，避免 readout 位置在 cache 命中时缺失 |
| chunked prefill | 后续范围 | 首版禁用，尚未设计跨 chunk 保存选项表示 |
| 图执行 | 后续范围 | 首版强制 eager，尚未验证 ragged metadata/output |
| 量化、动态 LoRA、多卡 | 后续范围 | 不属于第一版验收组合 |
| Qwen3.5 | 后续范围 | 当前注册和制品架构仅针对 Qwen3 causal decoder |
| 生成/流式、图像、MTP、PD 分离 | 非目标 | KEV 专用实例拒绝生成请求 |

状态矩阵不把服务可启动、dummy 通过或 synthetic 数值通过解释为真实权重验收通过。
