# 训练日志目录

点击 Unity 界面的“开始训练”后，程序会为这次点击分配唯一 `run_id`。每个 episode
结束时，日志在本目录累计写入 CSV，并重新生成 SVG；程序重启后历史行仍保留。

> 无参数入口中，Unity“开始训练”现在启动混合 SAC 自训练并实际更新权重；显式
> checkpoint、显式模式或 `--validate-only` 仍只做确定性运行。当前单回合默认最多
> 5000 个控制步或 600 s；旧 `MAX_EPOCH` 不变，自训练目标从累计 1000 回合开始。

## 生成文件

| 文件 | 内容 |
|---|---|
| `training_runs.csv` | 每次点击的 `run_id`、开始时间、episode 数和成功数 |
| `training_episodes.csv` | 每个 episode 的总步长、是否完成、到终点总步长、时长和停止原因 |
| `waypoint_steps.csv` | 每个 episode 的 13 个目标点累计步长、分段步长和最小距离 |
| `training_total_steps.svg` | 横轴为全局 Episode，纵轴为本回合总步长 |
| `waypoint_cumulative_steps.svg` | 13 条曲线分别表示抵达各目标点时的累计步长 |
| `waypoint_segment_steps.svg` | 13 条曲线分别表示相邻目标点之间使用的步长 |
| `self_training_episodes.csv` | 每个阶段回合的代次、reward、步数、成功、安全计数、loss 和训练步 |
| `self_training_generations.csv` | 每代父／候选 SHA、评估结果、晋级结果与原因 |
| `self_training_reward.svg` | 各阶段回合累计 reward 曲线 |
| `self_training_success_rate.svg` | 跨点击累计成功率曲线 |
| `self_training_total_steps.svg` | 自训练各阶段回合总步数曲线 |
| `self_training_losses.svg` | actor loss 与 critic loss 曲线 |
| `self_training_state/` | 最近一个完整回合的内容寻址续训状态；运行产物不入库 |

未到达的目标点在 CSV 中保留为空值，在 SVG 中显示为曲线断点，不会错误写成 0。
CSV 使用 `run_id` 区分多次点击，并用 `global_episode` 提供跨点击的连续横轴。
`global_episode` 和步长计数从 1 开始；为保持样例界面兼容，单次点击内部的
`episode` 字段仍从 0 开始。`total_steps_to_goal` 只在完成 13 点时写值，失败或
中止回合保持为空。步长表示实际执行的控制周期数，不等同于秒数或 SAC 更新次数。

自训练 CSV 使用持久化 `session_id`、`generation` 和 `stage` 区分 95/5/20/5 阶段；
半回合在停止或 Reset 时丢弃，不会推进状态。UI 中 `E` 为累计训练回合，`Step` 为
当前步，`Score` 为累计 reward，`Loss` 为最新 critic loss，`MaxE` 为当前累计目标。

SVG 采用白底、灰黑线、实线／虚线／点线和衬线字体，直接由 Python 标准库输出，
无需安装 Matplotlib。坐标轴会按全部实际数据自动缩放，5000 不是硬编码的显示上限。
CSV/SVG 是本机运行产物，已由 `.gitignore` 排除；本说明文件保留在版本库中。

## 聚焦验证记录

- [`training_logging.tdd.md`](./training_logging.tdd.md)：CSV/SVG 日志契约。
- [`zero_clearance_5000_steps.tdd.md`](./zero_clearance_5000_steps.tdd.md)：零额外净空、5000 步和 600 s 契约。
- [`point4_point5_slow_turn.tdd.md`](./point4_point5_slow_turn.tdd.md)：上一版点 3 后减速及第 4→5 点低速转弯历史证据。
- [`points3_to5_conservative_reset.tdd.md`](./points3_to5_conservative_reset.tdd.md)：当前点 3→4→5 保守控制及已撤回三次延迟复位的历史证据。
- [`planning_deferred_retry.tdd.md`](./planning_deferred_retry.tdd.md)：当前规划延迟零控制重试与 600 s 超时边界。
- [`self_training.tdd.md`](./self_training.tdd.md)：混合自训练、续训、v5 晋级与新增日志契约。
