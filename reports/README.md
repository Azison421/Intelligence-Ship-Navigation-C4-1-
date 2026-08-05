# 验证与运行证据

当前实现不再向旧版 CSV/SVG 追加数据。正式证据源分为三类：

- `artifacts/logs/forward-control-v2-*.json`：Unity 双向油门/舵效标定。
- `artifacts/logs/national_test_training_progress.json`：最近一个离线训练或评估回合的
  小型进度快照。
- `artifacts/logs/national-test-runtime-*.jsonl`：Unity 逐周期遥测，包含策略动作、
  可达 mask、安全 mask、最终 `NavigationStatus` 百分比命令和只读设备反馈。

`tools/export_training_reports.py` 将上述 V6 JSONL 单向投影为
`national_test_v6_*.csv/svg`。报表不是恢复源，也不会反向影响训练或 100 ms 控制循环；
旧报表继续保留但不迁移、不追加。

可恢复网络、优化器和 V3 回放位于 `artifacts/checkpoints`。旧 checkpoint、旧 Unity
截图和旧日志保留为历史证据，但当前 V6 加载器不会读取它们。

## 当前文档

- [`self_training.tdd.md`](./self_training.tdd.md)：V6 训练门、Unity 适应和 5/5 晋级
  规则。
- [`training_logging.tdd.md`](./training_logging.tdd.md)：当前 JSON/JSONL 遥测字段与
  边界。

已删除 v35–v37 的点位补丁、零净空和 `PLANNING_DEFERRED` TDD 报告；相应代码路径
已经不存在，继续保留活动文档会误导运行人员。历史内容仍可从 Git 读取。

离线测试、合成回放和 Markdown 审核都不能替代真实 ROS/Unity 验收。
