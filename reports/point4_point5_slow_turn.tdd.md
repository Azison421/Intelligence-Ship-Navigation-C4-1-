# 第 4→5 点低速转弯修复 TDD 证据

> 历史状态：本文冻结上一版 `0.1/0.15` 慢转候选的证据。Unity 随后出现持续
> `PLANNING_DEFERRED`，当前默认候选进一步收紧点 3→4 舵角并把点 4→5 舵角降为
> 0.12。后续曾加入的三次延迟复位现已撤回；当前证据见
> [`points3_to5_conservative_reset.tdd.md`](./points3_to5_conservative_reset.tdd.md) 和
> [`planning_deferred_retry.tdd.md`](./planning_deferred_retry.tdd.md)。

日期：2026-08-04  
来源：用户提供的 Unity 截图与现象说明；截图显示船穿过第 4 点后朝第 5 点转弯时，
会外飘碰到两点之间的浮标。

## 范围与成功标准

- Informed RRT*、SAC 权重、13 点顺序、路线代价和 0 m／0 m 安全 profile 不变。
- 第 3 点后的点 4 复合航段只使用不超过 0.1 的正向油门。
- 点 4 和西侧 handoff 的离线模型速度都不超过 0.15 m/s。
- 西侧 handoff 到点 5 使用 `Control(0.1, 0.15)`；在 0.4 m 船体碰撞模型下，
  该子航段连续有效且净空大于 0.35 m。
- 新配置只绑定到新的 `unity_test` 候选；旧 v37 与上一版零净空显式路径保持原
  `0.4/0.2` 局部控制。

## RED

先加入规划器、manifest profile、候选入口和 CLI 默认路径用例，再运行这 4 个点名
目标。结果为：

```text
4 failed in 10.69s
```

失败原因均符合预期：规划器不接受局部油门／转弯参数，运行时缺少机动 profile
解析器，新候选尚不存在，CLI 仍选择上一版零净空候选。

## GREEN

最小实现完成后，原 4 个 RED 目标结果为：

```text
4 passed in 21.21s
```

随后把运行时断言扩展到实际控制器，验证候选 manifest 参数确实进入点 4 规划，而
不只是能够被解析：

```text
1 passed in 22.23s
```

最后运行 10 个新旧点 4–5、候选加载和入口兼容目标。第一次运行有 1 个旧测试失败：
它把 `plan_clearance_turn()` 拼接的点 5→6 出口净空也计入“点 4→5 转弯”断言。
该测试被拆成“点 4→5 大于 0.3 m”和“完整轨迹大于地图要求”两个真实边界；生产
逻辑未因此改变。最终结果为：

```text
10 passed in 27.56s
```

使用当前候选 manifest 中的实际前向标定再次输出确定性离线指标：

```text
approach_max_throttle=0.100
point4_speed_mps=0.126813
handoff_speed_mps=0.126813
approach_min_clearance_m=0.408823
point4_to_5_max_speed_mps=0.126813
point4_to_5_min_clearance_m=0.421864
point5_reach_time_s=20.2
```

## 测试规格

| 保证 | 用例 | 类型 | 结果 |
|---|---|---|---|
| 点 3 后至 handoff 的正向油门不超过 0.1，点 4 与 handoff 速度不超过 0.15 m/s | `test_point_four_slow_profile_limits_speed_through_point_five` | 规划器单元／集成 | PASS |
| 点 4→5 使用 `0.1/0.15`，子航段船体净空大于 0.35 m | 同上 | 动力学与碰撞集成 | PASS |
| 旧 manifest 回退原机动，新 manifest 解析低速机动，非法参数失败关闭 | `test_runtime_maneuver_profile_keeps_legacy_defaults_and_parses_slow_turn` | 单元 | PASS |
| 新候选复用 v37 权重、仅允许 `unity_test`，且实际控制器生成低速轨迹 | `test_zero_clearance_candidate_reuses_v37_weights_and_is_unity_only` | 资产／运行时集成 | PASS |
| CLI 无参数入口选择新的低速候选，显式 `live` 仍选择正式默认路径 | 两个 `test_official_main_entry_*` 点名用例 | 入口兼容 | PASS |
| 旧点 4 复合交接与旧点 4→5 默认转弯继续满足原地图安全要求 | 两个旧规划／运行时点名用例 | 兼容 | PASS |

## 资产与验证边界

- 新文件为 `national_test_sac_v37_zero_clearance_slow_turn_unity_test.pt`；其权重字节
  与旧 v37、上一版零净空候选相同，manifest 独立且 readiness 均为 false。
- 三份候选的 SHA-256 都是
  `ba9243d5db3d3375a974373e48f617308b37bce7712c233288a034c0f5beb271`。
- 7 个相关 Python／测试文件通过 `py_compile`；候选资产 preflight 通过；7 份项目
  Markdown 相对链接检查为 0 个断链；`git diff --check` 通过（只有现有换行提示）。
- 没有运行全量 pytest，也没有重跑 13/13 零净空离线回放。
- 为遵守“不跑全量回归”的边界，没有生成全项目覆盖率报告；本次只报告上述点名
  行为与静态检查，不宣称达到全项目 80% 覆盖率。
- 没有启动、点击或操纵 Unity；没有连接 ROS、Simulink 或 MATLAB。
- 本报告只证明聚焦离线契约，不证明真实 Unity 船模已经避开第 4→5 点浮标。
- 未创建 TDD checkpoint commit；用户没有要求提交，修改保留在当前工作树。
