# 环境与基线核对报告 (Task 1.3 & 1.5)

## 1. 代码与仓库版本
- **KEV 原参考代码** (jaredpalmer/kev): `90990a5fac2995b9faa3190f7d437e84f2067768`
- **vllm-upstream (当前环境)**: `568afb3a13806beb53bb2e6bd518269357b237c0` (匹配 v0.26.0)
- **vllm-ascend (当前环境)**: `9aebd4dd8f3379f0d171bbdc4ec87544cd6aeb3c` (匹配 v0.26.0rc)

## 2. 模型制品版本
- **模型库**: `jaredpalmer/kev-4b`
- **Revision**: `qwen3` (`c4bfa11b0dc07691884f2d97f1c4c4c05c92e416`)
- **基座模型**: `Qwen/Qwen3-4B-Base` (Revision `906bfd4b4dc7f14ee4320094d8b41684abff8539`)
- **Head 元数据**: 见 `provenance.json` 和 `training_config.json`

## 3. 交付物
- 上游基线 Commit: `568afb3a13806beb53bb2e6bd518269357b237c0`
- Patch 路径: `evidence/vllm_upstream_kev.patch`
- Patch SHA256: 见 `evidence/vllm_upstream_kev.patch.sha256`
