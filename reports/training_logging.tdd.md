# National_Test V6 遥测契约

日期：2026-08-06

## 离线训练进度

`tools/train_fixed_map_sac.py` 每个完整回合更新
`artifacts/logs/national_test_training_progress.json`。字段包括阶段、代次、训练回合、
块内进度、评估次数/通过数、起始/结束航点、本局实际完成数、碰撞/超时、安全干预、
净空、停止原因和 reward。同一内容逐局追加到
`national-test-offline-training-*.jsonl`，进度 JSON 只保留最新一局；恢复训练只读取
严格的 `national-test-self-training-state-v6` 状态。V6 同时保存训练器和 PyTorch
随机状态，使暂停续训与不中断训练保持同一随机轨迹。

训练局还记录每局尝试/实际应用的更新数、critic loss、actor loss、actor 目标、
行为克隆 loss、温度 alpha 和策略熵。这些字段用于判断“程序执行了更新”与“策略
质量提高”是否同时发生，不能用单一 reward 曲线替代。第 101 局后的
`actor_objective=SAC_WITH_DAGGER` 表示 critic 仍按 SAC 学习实际动作，而 actor
同时使用当前策略状态上的专家标签；它不表示运行时存在专家 fallback。

`full_route` 区分完整路线与走廊分段扰动出生；只有 `gate_eligible=true` 的确定性
`OFFLINE_EVAL` 局可以计入 20/20。训练局从后段出生后偶然完成 13 点不算晋级。

运行 `python tools\export_training_reports.py` 可把当前 V3 JSONL 导出为 V6 episode
CSV。重复传入 `--offline-log` 会按训练局/评估局身份去重并合并续训日志。reward、
steps 和完成航点图只画完整路线训练局；学习诊断图显示完整路线 MA20 完成率、更新数、
critic/actor loss、alpha、确定性评估通过率及连续通过数。导出是只读投影，不参与
模型门控。

当前合并证据位于 `reports/national_test_v6_stability/`：CSV 包含 399 个成功落盘的
训练回合和 23 个评估回合，其中末尾 20 个正式评估回合全部完成 13 点。第 347 个
训练回合从完整状态恢复但缺少回合日志，导出器保持缺口而不推算或补写。

## Unity 运行遥测

`FixedMapNavigationService` 为每次进程创建一个
`national-test-runtime-YYYYMMDD-HHMMSS.jsonl`。每个控制周期记录：

- `policy_action`：SAC 提议；
- `reachability_mask`：舵角换向约束；
- `safe_action_mask`：预测安全结果；
- `executed_action` 和最终 `NavigationStatus` 油门/舵角百分比；
- `device_feedback`：只读 `DeviceStatus` 回读；
- 航点索引、目标距离、安全干预、停止原因和周期耗时。

控制异常、操作员中止和 episode 结束均单独记录。Unity 适应/验收的 `episode_end`
还记录阶段、完成结果、累计验收通过数以及适应局的更新数和 loss。使用
`--runtime-only --runtime-log <日志> --output-dir <新目录>` 会只生成 Unity 周期与
episode CSV/SVG。日志不包含 ROS 主机、设备标识或其他连接凭据。

计分回合结束后还会记录 `unity_reset_complete` 或 `unity_reset_failed`，用于区分
策略失败、`MOTION_STALLED`、复位失败和操作者尚未重新开始训练。

## 边界

- 单周期预算 100 ms；超限立即归零且本局不计训练门成绩。
- 单局墙钟上限 600 s。
- 操作员中止和输入陈旧保存安全前缀，但不施加终止惩罚，也不计晋级成绩。
- 只有真实 Unity 5/5 验收可以产生比赛默认 checkpoint。
