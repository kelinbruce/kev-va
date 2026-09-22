# 任务 5.3：确定性答案后处理证据

实现位置：`vllm-upstream/vllm/entrypoints/pooling/systemone/responses.py`。

实现先校验未舍入概率的数量、有限性、范围和逐题归一化，再计算 `noul`、`choice` 和 `score` 答案。choice 胜出项、score 期望及 confidence 均使用原始概率；并列最大概率按输入顺序选择；最后才保留两位小数。

验证结果：

```text
pytest --confcutdir=tests/entrypoints/pooling/systemone \
  tests/entrypoints/pooling/systemone -q
15 passed

固定 KEV to_answers 对比：三类型输出 exact
ruff check：通过
ruff format --check：6 files already formatted
```

测试包含规范 golden 值、舍入后并列但原始概率不同、原始概率精确并列、K=1 confidence，以及长度错误、NaN、越界值和未归一化分布。
