# Train 按钮 CSV/SVG 日志 TDD 证据

日期：2026-08-04  
来源：根据本轮用户需求直接形成，没有外部计划文件。

## 用户路径

1. 用户点击 Unity“开始训练”后，每次点击获得独立 `run_id`。
2. 每个 episode 结束时，用户可以从 CSV 读取总步长、到终点总步长，以及到达
   13 个目标点时的累计／分段步长。
3. 用户多次点击训练后，历史记录继续追加，SVG 使用全部历史 episode 自动重绘。
4. 未到达的目标点显示为空值和曲线断点，不得伪造为 0 步。

## RED

命令：

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_training_reports.py
```

结果：测试收集失败，错误为
`ModuleNotFoundError: No module named 'usvlib4ros.navigation.training_reports'`。
失败原因正是报告模块尚未实现，而不是测试语法或环境错误。

## GREEN 与覆盖率

| 保证 | 用例／命令 | 类型 | 结果 |
|---|---|---|---|
| 两次训练点击跨进程格式累计为连续 global episode | `test_report_logger_appends_runs_and_renders_step_curves` | 单元 | PASS |
| 3 份 CSV 与 3 张灰度 SVG 均生成 | 同上 | 单元 | PASS |
| 目标累计步长严格递增，错误数据被拒绝 | `test_report_logger_rejects_non_monotonic_waypoint_steps` | 单元 | PASS |
| Train UI 服务链在 episode 后写报告 | `test_train_button_episode_is_written_to_reports` | 集成 | PASS |
| 既有样例入口与生命周期保持兼容 | `tests/test_sample_entry_compatibility.py` | 集成 | 13 PASS |

最终点名验证：

```text
tests/test_training_reports.py + tests/test_sample_entry_compatibility.py
16 passed in 2.17s
```

环境没有安装 `coverage.py`，因此使用 Python 标准库 `trace` 对新模块执行相同的
3 个用例；结果为 410 个可执行行中覆盖 383 行，行覆盖率 **93.41%**。

测试生成的 `waypoint_cumulative_steps.svg` 又经本机无界面浏览器渲染为 PNG 做
视觉检查：坐标、图例、灰黑线型、单点和缺失值断点均正常。预览使用合成数据，未
写入正式 `reports` CSV/SVG。

## 边界

- 严格没有运行全量 pytest。
- 没有启动 Unity、ROS 或 MATLAB；UI 接入由隔离服务测试验证，真实点击结果仍需
  用户下一次 Unity 运行产生。
- 当前 UI“训练”是闭环 episode，不会在线更新 SAC 参数；日志步长不能解释为梯度
  更新次数。
- 工作树在本轮开始前已有其他未提交修改，因此没有创建 TDD checkpoint commit；
  RED/GREEN 证据保存在本文件中。
