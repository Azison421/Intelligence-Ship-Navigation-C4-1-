# 第 3→4→5 点保守控制与已撤回规划卡死复位 TDD 历史证据

日期：2026-08-04  
来源：用户提供的 Unity 终端截图与运行反馈。终端在第 5 个目标附近持续输出
`Action: None, Reason: PLANNING_DEFERRED, Point: 4, Distance: 1.70`，回合不再前进。

> 后续修订：Unity 实际运行表明一个规划控制步可能耗时数秒，连续 3 个步长不能
> 等价为固定的 0.3 s。三次延迟看门狗已撤回；当前行为是发布零控制并在同一回合
> 继续重试，证据见 [`planning_deferred_retry.tdd.md`](./planning_deferred_retry.tdd.md)。
> 点 3→4→5 保守控制保持不变。下文保留首次实现的 RED／GREEN 历史。

## 范围与约定

- Informed RRT*、SAC 权重、13 点顺序、路线代价、0 m／0 m 安全 profile 不变。
- 用户点号按 1 开始；运行时 `mission_index=3` 对应第 4 点，`mission_index=4`
  对应第 5 点。
- 第 3→4 段限制为 `油门≤0.1、|舵角|≤0.1`。
- 第 4→5 段保持油门 0.1，并采用离线能到达第 5 点的最小检查值舵角 0.12。
  舵角 0.1 的离线轨迹因转弯半径过大而无法到达，因此没有强行使用。
- 安全反向制动 `Control(-0.4, 0.0)` 保留；它只用于超速制动，不属于前进油门。
- 历史实现（现已撤回）：连续 3 个控制周期返回 `PLANNING_DEFERRED` 时，曾以
  `PLANNING_DEFERRED_RESET` 结束当前回合。
- 新参数只绑定到新的 `unity_test` 候选；旧 v37、旧零净空和上一版慢转路径保留。

## RED

先增加 5 个聚焦目标，覆盖规划器舵角包络、manifest profile、候选默认入口和回合
复位看门狗。实现前结果为：

```text
FFFFF
5 failed in 10.53s
```

失败原因分别为：`plan_fixed_leg()` 不接受舵角上限、运行时 profile 没有
`approach_rudder_cap`、默认 checkpoint 仍为上一版、CLI 仍解析到上一版，以及服务
没有 `PLANNING_DEFERRED_RESET_STREAK`。

## GREEN

最小实现后首次运行有 4 项通过，剩余 1 项使用了上一版慢转轨迹的经验断言
`approach_min_clearance > 0.35 m`。新保守轨迹实际为 0.336840 m，仍高于该段明确的
0.2 m 规划要求且没有船体接触，因此测试边界修正为 `>0.3 m`；生产路线没有为此
修改。相同 5 个目标最终结果为：

```text
.....
5 passed in 19.65s
```

随后运行 13 个新旧点 4–5、候选加载、规划延迟和样例入口兼容目标：

```text
.............
13 passed in 25.29s
```

没有运行全量 pytest。

## 确定性离线指标

使用候选 manifest 对应的 v37 前向标定和 0 m 额外净空再次计算：

```text
approach_max_throttle=0.100000
approach_max_abs_rudder=0.100000
point4_speed_mps=0.126813
handoff_speed_mps=0.126813
approach_min_clearance_m=0.336840
point4_to_5_max_speed_mps=0.126813
point4_to_5_min_clearance_m=0.535599
point5_reach_time_s=25.4
```

这些是简化动力学与静态 sidecar 下的离线值，不是 Unity 实测。

## 候选与兼容性

当前无参数 CLI 候选为：

```text
artifacts/checkpoints/national_test_sac_v37_zero_clearance_conservative_345_unity_test.pt
```

它与上一版慢转候选的二进制 SHA-256 均为：

```text
ba9243d5db3d3375a974373e48f617308b37bce7712c233288a034c0f5beb271
```

独立 manifest 的 `clearance_maneuver_profile` 记录 `0.1/±0.1` 和 `0.1/0.12`，
`unity_test_only=true`、`offline_ready=false`、`live_ready=false`。上一版 manifest
没有 `approach_rudder_cap` 时自动回退为 1.0，以保持显式旧路径行为。

## 测试规格

| 保证 | 测试 | 类型 | 结果 |
|---|---|---|---|
| 第 3→4 段油门和舵角均受保守包络限制，且第 4→5 段可达、连续无碰撞 | `test_points_three_to_five_conservative_profile_limits_controls_and_speed` | 规划集成 | PASS |
| 新字段可解析，旧 manifest 保持原行为，非法值失败关闭 | `test_runtime_maneuver_profile_keeps_legacy_defaults_and_parses_slow_turn` | 单元 | PASS |
| 新候选复用同一权重、仅允许 `unity_test`，运行时轨迹实际使用小舵角 | `test_zero_clearance_candidate_reuses_v37_weights_and_is_unity_only` | 运行时集成 | PASS |
| 无参数入口选择当前候选，显式 `live` 仍使用严格入口 | `test_official_main_entry_defaults_to_v37_conservative_345_candidate` 等 | 入口兼容 | PASS |
| 历史三次规划延迟看门狗 | `test_unity_episode_uses_5000_step_and_600_second_limits` | 服务集成 | REMOVED |

## 已知边界

- 没有运行覆盖率命令：用户明确禁止全量回归，本次只保留点名测试证据，不能声称
  全项目覆盖率达到某个百分比。
- 没有重跑 13/13 零净空离线回放，也没有启动 Unity、ROS、Simulink 或 MATLAB。
- 本文的自动复位结果只属于已撤回的历史实现，不代表当前运行行为。
- checkpoint 目录被 `.gitignore` 排除；本机候选和 manifest 已创建，但不会出现在
  普通 `git status` 中。
