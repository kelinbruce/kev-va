# 验收证据

此目录用于保存 `add-kev-qwen3-decision-serving` 变更在重新实施过程中产生的可复现证据。

当前分支从未实施状态开始。只在对应工作实际完成并验证后更新任务状态；CPU/dummy
结果不能代替 Ascend 真实权重验收。

- `task_1_3_source_report.md`：固定来源、revision 与文件清单。
- `task_1_5_8_1_delivery.md`：双仓库 checkout 与可应用上游 patch。
- `task_2_1_2_3_artifact_tooling.md`：离线导出/验证工具及拒绝测试。
- `task_3_1_3_3_5_1_5_7_5_8_6_1.md`：模型、API 与运行配置实现。
- `task_3_2_6_2_6_3_runtime.md`：进程边界、readiness 与 metrics。
- `task_4_1_4_2_renderer.md`：renderer、token 与索引。
- `task_4_3_4_4_pointer.md`：pointer 数值单元测试。
- `task_4_5_5_2_5_6_scheduling.md`：batch、限额、调度和生命周期。
- `task_5_3_postprocessing.md`：确定性后处理。
- `task_8_3_tutorial.md`：中文教程和文档 lint。
- `task_8_4_status_matrix.md`：环境、dummy、真实权重、数值和功能状态矩阵。
- `task_8_2_acceptance_entrypoints.md`：四份 spec 的测试映射、真实权重配置与
  benchmark 入口。
- `task_8_5_verification.md`：最终回归、格式检查、环境阻塞与 spec/evidence 审计。
