# 任务 2.1、2.3：模型制品导出与拒绝校验

## 实现范围

- `tools/kev_artifact.py` 在独立 Python 环境中读取固定来源清单，先核对
  `adapter_config.json`、`adapter_model.safetensors` 和 `head.pt` 的 SHA256。
- 只接受标准 Qwen3 LoRA；拒绝 Qwen3.5、DoRA、RS-LoRA、QA-LoRA、转置
  fan-in/fan-out、per-module rank/alpha、训练 token embedding 和
  option-isolation。
- LoRA 增量以 FP32 计算，再将全部 backbone 浮点张量转换为显式指定的
  `bfloat16` 或 `float16`；pointer head 的 weight/bias 保持 FP32。
- `head.pt` 使用 `torch.load(weights_only=True)`，避免执行 checkpoint 中的
  任意 pickle 对象。
- 导出清单记录固定来源 revision、原始训练文件 SHA256、导出参数、Python/
  Torch/Safetensors/Transformers 版本、结构 token ID 和全部输出文件 SHA256。
- 导出先在同目录临时路径完成全量校验，再替换最终目录；失败时清理临时文件，
  即使使用 `--force` 也保留已有制品。
- 校验器逐项核对清单与实际文件、不可变 revision、Qwen3 架构、配置一致性、
  shard index、张量集合/形状/dtype/有限性、正温度以及 tokenizer 结构 token。

## 可复现检查

运行环境：项目根目录 `.venv`；该环境与 Ascend 服务运行依赖分离。

```text
$ .venv/bin/ruff check tools/kev_artifact.py tests/test_kev_artifact.py
All checks passed!

$ .venv/bin/ruff format --check tools/kev_artifact.py tests/test_kev_artifact.py
2 files already formatted

$ .venv/bin/python -m pytest -q tests/test_kev_artifact.py
....................                                                     [100%]
20 passed in 1.37s

$ .venv/bin/python tools/kev_artifact.py --help
usage: kev_artifact.py [-h] {export,validate} ...
```

测试使用微型本地 safetensors 制品，不访问网络、不下载模型，覆盖标准 PEFT
键名映射和端到端导出，以及校验值损坏、路径穿越、错误架构、可变 revision、
缺失/错形/非有限 head、非正温度、option-isolation、无效结构 token、未校验
权重文件等拒绝路径。

## 明确保留的未完成项

- 任务 2.2 尚未完成：按用户要求不下载模型，本轮未产出完整真实权重制品。
- 任务 2.4 尚未完成：未执行真实模型 FP32 合并前后概率误差和运行 dtype 误差。
- 任务 2.5 尚未完成：未用完整真实制品执行离线 vLLM 重复加载。
