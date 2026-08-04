# 规划延迟同回合重试 TDD 证据

日期：2026-08-04  
来源：用户的 Unity 实际运行反馈及 `reports/training_episodes.csv` 中既有回合记录。

## 问题与边界

- Unity 中一次规划调用可能耗时数秒，因此“连续 3 个控制步”不是可靠的 0.3 s 计时。
- 原看门狗会在第 4 次本可完成规划前结束回合，阻断第 3 点之后的后续计算。
- 当前删除按连续次数结束回合的状态；`PLANNING_DEFERRED` 仍发布零控制，并在同一
  回合的下一次调用继续尝试。
- 既有 600 s 墙钟超时、5000 步上限、碰撞和停止请求保持不变。
- 第 3→4→5 点的小油门／小舵角 profile、Informed RRT*、SAC 权重和 13 点顺序不变。

## RED

先把服务集成测试改为：前 3 次返回 `PLANNING_DEFERRED`，第 4 次返回
`MISSION_DONE`；期望回合成功完成。旧实现结果为：

```text
F
1 failed in 3.75s
```

失败发生在第 3 次延迟后，证明看门狗提前返回，未给第 4 次规划机会。

## GREEN

仅删除 `PLANNING_DEFERRED_RESET_STREAK`、延迟计数和对应提前返回分支；普通
`PLANNING_DEFERRED` 的零控制行为没有改变。相同测试结果为：

```text
.
1 passed in 3.49s
```

随后只运行规划延迟、5000 步／600 s、当前候选加载及第 3→4→5 点保守控制的
6 个关联目标：

```text
......
6 passed in 19.79s
```

## 测试规格

| 保证 | 测试 | 类型 | 结果 |
|---|---|---|---|
| 连续 3 次规划延迟后，第 4 次仍可完成回合 | `test_unity_episode_uses_5000_step_and_600_second_limits` | 服务集成 | PASS |
| 5000 步上限与 600 s 墙钟保护保持有效 | 同上 | 服务集成 | PASS |

## 验证边界

- 只运行点名离线测试、语法和文档检查；没有运行全量 pytest 或覆盖率。
- 没有启动或操作 Unity、ROS、Simulink 或 MATLAB；真实 Unity 行为由用户复测。
