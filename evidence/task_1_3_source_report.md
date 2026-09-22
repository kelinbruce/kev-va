# 任务 1.3：固定来源与 checkpoint 核对

## 固定版本

- KEV 参考代码：`jaredpalmer/kev@90990a5fac2995b9faa3190f7d437e84f2067768`
- KEV Qwen3 权重：`jaredpalmer/kev-4b@c4bfa11b0dc07691884f2d97f1c4c4c05c92e416`（浮动引用 `qwen3` 的本次解析值）
- 基座：`Qwen/Qwen3-4B-Base@906bfd4b4dc7f14ee4320094d8b41684abff8539`

完整文件 SHA256 和结构化元数据见 [qwen3-checkpoint.json](sources/qwen3-checkpoint.json)。所有 16 个发布文件均从固定模型 revision 下载后重新计算 SHA256；三个 LFS 文件的结果同时与仓库 tree API 中的 LFS OID 一致。

## Head 核对

`head.pt` 包含 `q.weight`、`q.bias`、`k.weight` 和 `k.bias`，均为有限 FP32 参数。hidden size 为 2560，head dimension 为 256。checkpoint 未保存显式 `temperature`；固定参考代码的 `CheckpointMeta.temperature` 默认值为 1.0，因此本次记录温度为 1.0，并保留该默认值来源。

## 来源缺口

发布制品的 `provenance.json` 记录训练代码 commit 为 `29d71c78368657b3a522729a01c748ea15272abc`。该 commit 是固定参考 commit `90990a5f` 的祖先，但 provenance 中记录的 20 个源文件 SHA256 在 `90990a5f` 均已发生变化。

因此，`90990a5f` 是本项目后续 renderer、API 和数值参考的固定代码版本，不能描述为发布权重的原始训练代码。后续任务必须使用真实权重参考一致性测试证明两者兼容，并在不通过时保留差异，不能忽略这个来源缺口。
