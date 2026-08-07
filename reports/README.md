# 验证与运行证据

当前实现不再向旧版 CSV/SVG 追加数据。正式证据源分为三类：

- `artifacts/logs/forward-control-v2-*.json`：Unity 双向油门/舵效标定。
- `artifacts/logs/national_test_training_progress.json`：最近一个离线训练或评估回合的
  小型进度快照。
- `artifacts/logs/national-test-runtime-*.jsonl`：Unity 逐周期遥测，包含策略动作、
  可达 mask、安全 mask、最终 `NavigationStatus` 百分比命令和只读设备反馈。

`tools/export_training_reports.py` 只接受当前
`national-test-offline-training-v3` JSONL，并单向投影为
`national_test_v6_*.csv/svg`。完整路线 reward/steps/航点图与随机分段训练分离，同时
输出 MA20 完成率、SAC 更新数、critic/actor loss、alpha、确定性评估通过率和连续
通过数。重复 `--offline-log` 可合并续训日志；`--runtime-only` 可把 Unity 周期、
episode 结果和适应 loss 单独输出到新目录。报表不是恢复源，也不会反向影响训练或
100 ms 控制循环；旧报表继续保留但不迁移、不追加。

`national_test_v6_offline_1754/` 是独立 1754 局图表证据分支。训练期间仅临时存在
`.training.jsonl`，完成后删除该工作日志，目录最终只保留 CSV/SVG。

`national_test_v6_stability/` 保存当前发布候选从第 1 局到第 400 局及正式评估的
合并 CSV/SVG。正式评估为 20/20；第 347 局在 Windows 原子替换冲突后从完整状态
恢复，但回合日志未成功落盘，因此 CSV 保留这一真实缺口，不补造数据。

可恢复网络、优化器、随机状态和 V3 回放位于
`artifacts/checkpoints/national_test_self_training_v6.pt`。旧 checkpoint、旧 Unity
截图和旧日志保留为历史证据，但当前 V6 加载器不会读取它们。

## 当前文档

- [`self_training.tdd.md`](./self_training.tdd.md)：V6 训练门、Unity 适应和 5/5 晋级
  规则。
- [`training_logging.tdd.md`](./training_logging.tdd.md)：当前 JSON/JSONL 遥测字段与
  边界。

已删除 v35–v37 的点位补丁、全局零净空和 `PLANNING_DEFERRED` TDD 报告；相应代码
路径已经不存在，继续保留活动文档会误导运行人员。当前仅第 11→12 点哈希绑定狭窄
区按实船极限使用 0 m 附加净空，常规区域仍为 0.2 m。历史内容仍可从 Git 读取。

离线测试、合成回放和 Markdown 审核都不能替代真实 ROS/Unity 验收。
