# 任务 8.2：测试配置与 benchmark 入口

该任务交付验收入口和覆盖映射，不宣称已执行真实权重或 NPU 测试。真实执行结果仍由
任务 7.1--7.7 约束。

## 交付物

- `configs/kev_real_weight_acceptance.json`：固定模型 revision、数据集位置、样本覆盖、
  数值阈值、生命周期场景和性能矩阵。
- `tools/kev_benchmark.py`：输出扫点计划，使用已核验 token 数的本地 state workload
  执行 SystemOne HTTP benchmark；记录 warm-up、样本数、P50/P95、吞吐、失败详情，
  并从 `/metrics` 计算实际输入 tokens/s。
- `tests/test_kev_benchmark.py`：配置防漂移、矩阵、请求形状、Prometheus 解析、原子
  响应失败和外部测量声明测试。
- 上游 vLLM 的 model/pooler/SystemOne UT 与进程边界集成测试。

`plan` 当前生成 10 个去重 case：基线，state tokens 128/1024/4096、问题数 1/4/16、
候选项数 2/8/32、并发 1/4/16 的单因素扫点，以及
`s1024-q4-k8-c4` 多因素组合。工具不下载 tokenizer 或模型；实际环境必须提供由固定
tokenizer 核验过的 workload。峰值 NPU 显存必须从 Ascend 运行时另行采集，结果中
在未提供时保持显式缺失。

## 四份 spec 覆盖映射

| Spec | CPU UT/ST | 真实环境配置/入口 |
|---|---|---|
| `kev-model-artifact` | `tests/test_kev_artifact.py` 覆盖不可变 revision、FP32 LoRA/head、非默认温度、损坏/缺失/错误架构、离线完整制品校验 | 配置固定三项 revision 与 artifact path；任务 2.2/2.4/2.5 运行真实导出 |
| `kev-decision-inference` | renderer、question 隔离、索引、ragged FP32 pointer、batch 重排/混排与进程边界测试 | parity/boundary JSONL、全部数值阈值、阈值动作和性能维度写入配置 |
| `systemone-decision-api` | protocol、golden 后处理、限额、未知模型、乱序、容量、失败、超时、断连、usage、generation 隔离测试 | 三类型真实 smoke、边界集、生命周期列表和 HTTP benchmark 请求入口 |
| `kev-ascend-runtime` | 启动 profile 拒绝、artifact/readiness gate、warm-up、logical/actual metrics 与取消不计成功吞吐测试 | 环境/制品状态矩阵、实际 metrics counter、单因素/多因素 benchmark；Ascend dummy/真实权重保持待执行 |

其中 reference/NPU 数值对比、业务阈值、真实 batch invariance 和环境失败复现由配置中的
冻结数据路径、阈值和覆盖字段约束；不能用 CPU 测试结果替代。

## 本地验证

```text
$ ./.venv/bin/python tools/kev_benchmark.py plan
# 10 cases; exit 0

$ ./.venv/bin/python -m pytest -q \
    tests/test_kev_benchmark.py tests/test_kev_artifact.py
............................                                             [100%]
28 passed

$ ./.venv/bin/ruff check tools/kev_benchmark.py tests/test_kev_benchmark.py \
    tools/kev_artifact.py tests/test_kev_artifact.py
All checks passed!
```
