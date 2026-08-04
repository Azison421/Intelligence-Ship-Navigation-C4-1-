# 训练日志目录

点击 Unity 界面的“开始训练”后，程序会为这次点击分配唯一 `run_id`。每个 episode
结束时，日志在本目录累计写入 CSV，并重新生成 SVG；程序重启后历史行仍保留。

> Unity 界面的“训练”在当前入口中表示运行一个闭环 episode，并不会在线更新 SAC
> 权重。CSV/SVG 记录的是 Unity 闭环步长，不应与离线 SAC 梯度更新次数混为一谈。

## 生成文件

| 文件 | 内容 |
|---|---|
| `training_runs.csv` | 每次点击的 `run_id`、开始时间、episode 数和成功数 |
| `training_episodes.csv` | 每个 episode 的总步长、是否完成、到终点总步长、时长和停止原因 |
| `waypoint_steps.csv` | 每个 episode 的 13 个目标点累计步长、分段步长和最小距离 |
| `training_total_steps.svg` | 横轴为全局 Episode，纵轴为本回合总步长 |
| `waypoint_cumulative_steps.svg` | 13 条曲线分别表示抵达各目标点时的累计步长 |
| `waypoint_segment_steps.svg` | 13 条曲线分别表示相邻目标点之间使用的步长 |

未到达的目标点在 CSV 中保留为空值，在 SVG 中显示为曲线断点，不会错误写成 0。
CSV 使用 `run_id` 区分多次点击，并用 `global_episode` 提供跨点击的连续横轴。
`global_episode` 和步长计数从 1 开始；为保持样例界面兼容，单次点击内部的
`episode` 字段仍从 0 开始。`total_steps_to_goal` 只在完成 13 点时写值，失败或
中止回合保持为空。步长表示实际执行的控制周期数，不等同于秒数或 SAC 更新次数。

SVG 采用白底、灰黑线、实线／虚线／点线和衬线字体，直接由 Python 标准库输出，
无需安装 Matplotlib。CSV/SVG 是本机运行产物，已由 `.gitignore` 排除；本说明文件
保留在版本库中。
