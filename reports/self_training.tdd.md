# National_Test SAC V6 门控证据

日期：2026-08-05

## 固定契约

- checkpoint：`national-test-sac-checkpoint-v6`
- 观测：`local-waypoint-observation-v3`，166 维
- 动作：`five-calibrated-controls-v3`，5 个百分比唯一的前进控制
- 回放：`national-test-replay-v3`
- 状态：`national-test-self-training-state-v2`
- 地图净空／激光急停：0.2 m／0.6 m

新训练从随机网络、空回放和指定 seed 开始，不读取任何 v37 权重或旧训练状态。

## 门控顺序

```text
100 局离线训练
  -> 20 局完整路线确定性评估（必须 20/20）
  -> 5 局 Unity 适应（允许更新）
  -> 5 局 Unity 验收（权重冻结，必须 5/5）
  -> PROMOTED
```

离线评估失败后进入下一 100 局块；累计 1000 个训练回合仍未达到 20/20 时明确进入
`TRAINING_GATE_FAILED`。Unity 验收失败数据进入 V3 回放，再返回离线训练；不会把
失败候选伪装为比赛默认模型。

## 已完成的现场证据

- ROS 管理节点只订阅 `NavigationStatus` 控制，并发布只读 `DeviceStatus`。
- Windows 与 ROS 时钟约有 15.5 s 偏差；命令时间已改为使用设备反馈时钟，避免
  被管理节点的 2 s 新鲜度门拒绝。
- 10% 油门经 `NavigationStatus` 单通道到达设备回读，停止后持续归零。
- 双向标定日志：`forward-control-v2-20260805-173657.json`。
- 标定动作：`(10,-10)、(10,-5)、(40,0)、(10,5)、(10,10)`；两侧舵效均成立。

## 离线环境修正

- 冻结地图和走廊在一个训练器生命周期内只编译一次。
- 出生扰动必须至少有一个通过同一 2 s 安全预测的动作。
- 走廊投影受当前比赛航点限制，避免 11→12 回头段提前吸附到下一支路。
- 安全零命令期间仍用已标定动力学传播惯性，不冻结船体位置。
- 三个 GRU 头使用 PyTorch `LayerNorm` 处理原始 V3 向量，避免 72 路量程让门控饱和；
  SAC 温度梯度按“熵高于目标则降低 alpha”的方向更新。
- 回放只保留最近 64 个完整 episode，既不跨 episode 取序列，也不让 1000 局门控
  状态无界膨胀。
- Windows 状态文件原子替换对短暂占用做有界重试；进度监控不再读取正在写入的
  PyTorch 状态。

## 验证边界

当前活动训练结果以 `artifacts/logs/national_test_training_progress.json` 和严格状态
文件为准。本文件不把训练局中的偶然完成当作 20/20，也不把离线门通过当作 Unity
5/5。未完成最后五局前，不得宣称存在比赛默认模型。
