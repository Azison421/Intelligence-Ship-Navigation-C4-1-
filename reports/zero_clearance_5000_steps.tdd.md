# Unity 零净空与 5000 步实验 TDD 证据

日期：2026-08-04  
范围：只验证零额外净空、0 m 激光接触阈值、候选门禁、5000 步／600 s 和日志兼容；
不改变算法、权重、13 点顺序或局部点位策略。

> 本文冻结的是第一版零净空候选的证据。随后针对 Unity 截图中的第 4→5 点碰撞
> 新增了独立低速机动候选；其 RED／GREEN 证据见
> [`point4_point5_slow_turn.tdd.md`](./point4_point5_slow_turn.tdd.md)。当前默认候选又
> 增加点 3→4 小舵角包络；后续加入的三次规划延迟复位现已撤回，证据见
> [`points3_to5_conservative_reset.tdd.md`](./points3_to5_conservative_reset.tdd.md) 和
> [`planning_deferred_retry.tdd.md`](./planning_deferred_retry.tdd.md)。

## 成功标准

- 旧 v37 manifest 缺少 `safety_profile` 时继续使用 0.2 m／0.6 m 基线。
- 新候选从 manifest 获得 0 m／0 m，只能由 `unity_test` 加载。
- 取消额外净空后仍保留 0.4 m 船体外形、预测碰撞和失效保护。
- Unity 控制循环与离线训练／评估默认最多 5000 步；墙钟及相关证据上限为 600 s；
  `MAX_EPOCH` 保持 4000。
- CSV 能记录 `total_steps=5000`，SVG 继续按数据自动缩放。

## RED

第一组先加入 8 个核心点名用例。命令只运行新用例，结果为：

```text
7 failed, 1 passed in 8.92s
```

失败分别锁定了旧默认候选名、3000 步默认值、缺失的运行时安全 profile、只能按
0.2 m 编译地图以及零净空候选尚不存在。SVG 自动缩放用例在实现前已经通过，证明
5000 并非图表硬编码上限，无需改动报告格式。

全项目定向搜索又发现两个单回合辅助脚本和晋级证据校验仍固定为 300 s，因此补了
两个独立 RED：

```text
test_bounded_episode_tools_use_600_second_limit
1 failed in 3.47s

test_promotion_requires_three_matching_passed_unity_validation_logs
1 failed in 2.69s
```

前者因辅助脚本没有 600 s 常量失败；后者使用 599 s 合法证据，因旧 300 s 门槛
失败。一次系统 Python 对含中文路径的导入失败属于环境问题，没有计入 RED；以上
证据均使用项目 `.venv` 复现。

## GREEN

最终新增功能点名组：

```text
10 passed in 21.08s
```

该组逐项覆盖：

| 用例 | 结果 |
|---|---|
| `test_runtime_safety_profile_keeps_legacy_defaults_and_parses_zero` | PASS |
| `test_zero_clearance_map_removes_buffer_but_keeps_footprint_collision` | PASS |
| `test_zero_clearance_runtime_only_stops_laser_at_contact` | PASS |
| `test_zero_clearance_candidate_reuses_v37_weights_and_is_unity_only` | PASS |
| `test_official_main_entry_defaults_to_v37_zero_clearance_candidate` | PASS |
| `test_unity_episode_uses_5000_step_and_600_second_limits` | PASS；模拟跑满 5000 步，并在 601 s 超时 |
| `test_bounded_episode_tools_use_600_second_limit` | PASS |
| `test_offline_evaluation_uses_competition_step_limit` | PASS |
| `test_report_logger_records_5000_steps_with_automatic_svg_scale` | PASS |
| `test_promotion_requires_three_matching_passed_unity_validation_logs` | PASS；599 s 证据可接受 |

另运行 10 个旧基线／入口／日志点名兼容用例，结果：

```text
10 passed in 13.90s
```

它们验证显式 `live` 仍选择正式默认 checkpoint、旧 profile 的 0.6 m 激光行为、
地图无效与激光接触时的失败关闭、空 Unity 路线 fallback、Train 触发和既有 CSV/SVG
追加行为。

资产完整性最终检查要求：旧／新 checkpoint SHA-256 同为
`ba9243d5db3d3375a974373e48f617308b37bce7712c233288a034c0f5beb271`；旧基线地图 hash
为 `1b71c938899e7fee3f6df93de260408a60d5c77f7719016e36f23353a4a1da1d`；零净空地图
hash 为 `ed04ab551655edbe8a104e392201695360a9cf01044a2cccfbc9535839da9f57`；路线 hash
保持 `2557e09d5bd881985ec2d16fd6d76498a6687f528e7da933f61d68cf3eb44058`。

最终静态检查结果：13 个改动 Python／测试文件全部通过 `py_compile`；6 份项目
Markdown 的相对链接检查为 0 个断链；`git diff --check` 通过。候选资产 preflight
也在不建立外部连接的情况下通过。

## 验证边界

- 没有运行全量 pytest，也没有重跑 13/13 零净空离线回放。
- 没有启动、点击或操纵 Unity；没有连接 ROS、Simulink 或 MATLAB。
- 新 manifest 的 `offline_ready`、`live_ready` 均保持 `false`，本证据不构成晋级。
- SVG/CSV 格式没有改变；测试数据只写到 pytest 临时目录。
- 未创建 TDD checkpoint commit；用户没有要求提交，修改保持在当前工作树供检查。
