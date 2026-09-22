# 任务 8.3：中文教程与索引

## 交付内容

- vLLM-Ascend 教程：
  `docs/source/tutorials/models/KEV-Qwen3-Decision.zh.md`
- 模型教程索引：`docs/source/tutorials/models/index.md`
- vLLM-Ascend commit：
  `0553039e12b671e37de5fbce414dd6a4503a3405`

教程覆盖离线制品导出与校验、首版启动参数、`noul`/`choice`/`score` 三种请求、
生成接口隔离、容量与 token 限额、稳定错误码、指标、真实验收要求和回滚步骤。
文档明确区分 CPU 测试、未验证的 CANN 9.0.1 环境以及尚未执行的 Ascend dummy/
真实权重验收，没有把未测指标标记为通过。所有示例均使用本地制品路径，不要求或
触发模型下载。

## 验证记录

```text
$ PRE_COMMIT_HOME=/private/tmp/kev-pre-commit-cache \
    ../.venv/bin/pre-commit run --files \
    docs/source/tutorials/models/index.md \
    docs/source/tutorials/models/KEV-Qwen3-Decision.zh.md
codespell................................................................Passed
typos....................................................................Passed
Check file name..........................................................Passed
Check suggestion........................................................Passed
```

完整默认 pre-commit 中的 `gitleaks` 因开发机缺少 `wget`、`check-symbolic-meta` 因
hook 环境缺少 `python` 而无法运行；它们不是文档内容失败。手动阶段的 markdownlint
随后单独执行并通过：

```text
$ PRE_COMMIT_HOME=/private/tmp/kev-pre-commit-cache \
    ../.venv/bin/pre-commit run markdownlint --hook-stage manual --files \
    docs/source/tutorials/models/index.md \
    docs/source/tutorials/models/KEV-Qwen3-Decision.zh.md
markdownlint.............................................................Passed

$ git diff --check
# exit 0
```

首次安装 markdownlint hook 时只下载了 Node/lint 工具链；未下载模型或权重。
