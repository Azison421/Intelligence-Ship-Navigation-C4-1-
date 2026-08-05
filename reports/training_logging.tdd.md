# National_Test V6 遥测契约

日期：2026-08-05

## 离线训练进度

`tools/train_fixed_map_sac.py` 每个完整回合更新
`artifacts/logs/national_test_training_progress.json`。字段包括阶段、代次、训练回合、
块内进度、评估次数/通过数、步数、完成航点、碰撞/超时、安全干预、净空、停止原因
和 reward。同一内容逐局追加到 `national-test-offline-training-*.jsonl`，进度 JSON
只保留最新一局；
恢复训练只读取严格的 `national-test-self-training-state-v2` 状态。

`full_route` 区分完整路线与走廊分段扰动出生；只有 `gate_eligible=true` 的确定性
`OFFLINE_EVAL` 局可以计入 20/20。训练局从后段出生后偶然完成 13 点不算晋级。

运行 `python tools\export_training_reports.py` 可把最新离线 JSONL 导出为 V6 episode
CSV，以及 reward、steps、完成航点 SVG。传入 `--runtime-log` 后还会把 Unity 控制周期
导出为扁平 CSV 和 cycle time／mission index SVG。导出是只读投影，不参与模型门控。

## Unity 运行遥测

`FixedMapNavigationService` 为每次进程创建一个
`national-test-runtime-YYYYMMDD-HHMMSS.jsonl`。每个控制周期记录：

- `policy_action`：SAC 提议；
- `reachability_mask`：舵角换向约束；
- `safe_action_mask`：预测安全结果；
- `executed_action` 和最终 `NavigationStatus` 油门/舵角百分比；
- `device_feedback`：只读 `DeviceStatus` 回读；
- 航点索引、目标距离、安全干预、停止原因和周期耗时。

控制异常、操作员中止和 episode 结束均单独记录。日志不包含 ROS 主机、设备标识或
其他连接凭据。

## 边界

- 单周期预算 100 ms；超限立即归零且本局不计训练门成绩。
- 单局墙钟上限 600 s。
- 操作员中止和输入陈旧保存安全前缀，但不施加终止惩罚，也不计晋级成绩。
- 只有真实 Unity 5/5 验收可以产生比赛默认 checkpoint。
