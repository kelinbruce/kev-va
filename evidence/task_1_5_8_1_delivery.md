# 任务 1.5（部分）、8.1：checkout 与上游 patch 交付

## 固定 checkout

| 项目 | revision | 状态 |
|---|---|---|
| 上游 vLLM 基线 | `568afb3a13806beb53bb2e6bd518269357b237c0` (`v0.26.0`) | patch 基线 |
| 上游 vLLM KEV 实施 | `dc7f73910b48c0d7ba2446bfcc3a1168fe716005` | 已测试代码 |
| vLLM-Ascend 原始基线 | `9aebd4dd8f3379f0d171bbdc4ec87544cd6aeb3c` | 尚无 KEV 后端改动 |
| vLLM-Ascend 交付 checkout | `0553039e12b671e37de5fbce414dd6a4503a3405` | 增加中文教程和索引 |

本地 `.venv` 的 `vllm` 实际解析到：

```text
/Users/zhangfan/project/kev-asend/vllm-upstream/vllm/__init__.py
```

`vllm_ascend` 在当前 macOS 开发环境未安装（`find_spec` 返回 `None`），因此任务
1.5 仍保持未完成：必须在实际 Ascend 容器中再次记录两个包的 import path，避免
把本地源码解析结果当成 NPU 运行时证据。

## Patch

- 文件：[`patches/vllm-kev-qwen3-v0.26.0.patch`](patches/vllm-kev-qwen3-v0.26.0.patch)
- 大小：104181 bytes
- SHA256：`60719a6eae47f458581ea04453bc80c0b7f5eca70ed8b3c1a4502b09a16cd9e9`
- 内容范围：固定 `v0.26.0` 基线至上述 KEV 实施 commit 的完整 binary-safe diff，
  包含实际运行代码和测试代码。

验证在临时 detached worktree 中从精确基线执行，没有修改工作区：

```text
$ git worktree add --detach /private/tmp/kev-vllm-patch-check 568afb3a1
HEAD is now at 568afb3a1 [CI/Build] Refresh tags before building macOS wheel (#49901)

$ git -C /private/tmp/kev-vllm-patch-check apply --check \
    evidence/patches/vllm-kev-qwen3-v0.26.0.patch
# exit 0

$ git diff --check 568afb3a1..dc7f73910
# exit 0
```

同时在实施 checkout 执行 `git apply --check --reverse` 通过，证明 patch 的终点
与当前已测试代码一致。临时 worktree 已删除。
