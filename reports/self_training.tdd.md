# National_Test SAC 自训练与自动迭代 TDD 证据

日期：2026-08-04  
范围：只验证混合自训练编排、SAC 更新、双回放、续训状态、v5 证据／晋级、入口兼容
和新增日志；不启动 Unity、ROS、Simulink 或 MATLAB，不运行全量回归。

## 1. 目标与安全边界

- 无参数 `main.py` 的 Train 触发混合自训练；显式 checkpoint、显式模式和
  `--validate-only` 保持确定性且不更新参数。
- 初始冠军为 v37 零净空保守 3→4→5 候选。每代顺序为 95 个离线训练、5 个 Unity
  探索、20 个离线评估、5 个 Unity 确定性验收；首次另有 5 个冠军 Unity 基线。
- 点 3→4 的硬上限为 `0.1/±0.1`，点 4→5 为 `0.1/±0.12`；0.4 m 船体、预测
  碰撞、数据新鲜度和安全监督最终否决权保留。
- 单回合仍为 5000 步／600 s。`PLANNING_DEFERRED` 只发布零控制并重试，不新增
  连续次数 Reset。

## 2. RED 与基线

实现前，新增契约首先锁定了以下缺口：Train 服务没有自训练状态机，不会调用 SAC
更新；没有可恢复的网络／优化器／回放状态；没有 v5 代次证据与活动冠军指针；无参数
入口的 checkpoint 权重哈希不会因 Train 改变。新增用例在相应模块和入口接线尚不存在
时失败，随后才实现生产代码。

本轮开始时还点名运行过既有离线基线：4 项通过，
`test_fixed_map_sac_trains_complete_safe_episode_and_saves_checkpoint` 在旧规划回放的
`mission_index=7` 超时失败。该慢规划用例不是本功能的 RED 目标，本轮没有为了得到
全绿而改写它，也没有运行全量回归。

## 3. 实现结果

### 状态机与续训

- `SelfTrainingCursor` 固定 5 基线和 95/5/20/5 阶段，并按 1000 个训练回合扩展目标。
- 每个完整回合后用内容哈希和原子指针保存网络、目标网络、四个优化器、温度、训练
  步、Python／Torch 随机状态、训练器随机状态、32／20 回合双回放池及阶段游标。
- 点击结束或进程停止时丢弃当前半回合；已完成回合已经推进游标并持久化。状态哈希、
  schema、配置或冠军哈希不匹配时失败关闭，不静默创建新会话。
- 非有限 loss 或更新异常时 SAC 恢复最近冠军、发布零控制；发生异常前已完整结束的
  回合仍先写入回放并保存进度。

### 学习与安全

- 每个离线训练回合执行 16 次 SAC 更新；每 5 个 Unity 探索回合后执行 80 次更新。
- 无 Unity 样本时批量为 8 条离线样本；有 Unity 样本后固定为 6 条离线加 2 条 Unity。
- v5 策略可从所有通过预测安全掩码的动作中选择；特殊段生成 5 个有差异且不越过
  profile 硬上限的候选。v4 继续使用原最小干预门禁。
- 新鲜 Unity 时间戳形成连续 RNN transition；重复、倒序和跨 Reset session 的样本
  不会重复写入。碰撞记录为可学习的终止负样本、立即零控制并使当代禁止晋级。

### checkpoint、晋级与日志

- 每代生成不可覆盖的 v5 checkpoint 和 manifest，包含父模型 SHA、训练谱系、完整
  profile、训练回合数、20/20 离线与 5/5 Unity 聚合证据和晋级判定。
- 只有两组评估均全完成且零安全事件，并在完成率相同条件下使 Unity 中位总步数至少
  改善 2%，才原子更新 `national_test_sac_active.json`。失败代恢复冠军网络／优化器，
  但保留纠错回放。显式 `--checkpoint PATH` 永远不跟随活动指针。
- 旧 3 份 CSV／3 张 SVG 不变；新增 `self_training_episodes.csv`、
  `self_training_generations.csv` 和 reward、成功率、总步数、actor/critic loss 4 张 SVG。

## 4. GREEN 聚焦验证

最终只运行了下列相关测试集合：

```text
tests/test_self_training.py
tests/test_recurrent_sac.py
tests/test_training_reports.py
tests/test_sample_entry_compatibility.py
tests/test_fixed_map_trainer.py 中 2 个自训练／5000 步点名用例
tests/test_fixed_map_runtime.py 中 5 个 v5／零净空／船体碰撞点名用例
```

结果：`61 passed in 16.93s`。

这些用例覆盖：

| 契约 | 证据 |
|---|---|
| 5 + 95/5/20/5 状态机、1000 回合扩展 | `test_default_self_training_schedule_is_exact_and_extends_in_1000_episode_blocks` |
| 6:2 混合批次、纯离线首批、真实权重／训练步／checkpoint 哈希变化 | `test_mixed_replay_batch_is_six_offline_and_two_unity_when_unity_exists`、`test_learner_uses_pure_offline_then_six_two_mixed_batches_and_updates_weights` |
| 完整训练状态与双回放持久化、损坏拒绝 | `test_training_state_round_trip_restores_network_optimizers_temperature_and_step`、`test_state_store_round_trips_only_complete_episode_state_and_rejects_corruption` |
| Unity 连续 RNN 样本、重复／过期过滤、碰撞负样本 | `test_unity_transition_recorder_is_continuous_ignores_duplicate_samples_and_marks_collision_negative` |
| 零净空、船体接触、激光接触、特殊段 5 候选硬上限 | `test_special_segment_has_five_distinct_candidates_inside_hard_limits` 及 4 个运行时点名用例 |
| 20/20、5/5、碰撞禁晋级、2% 比较 | `test_promotion_requires_both_strict_gates_and_two_percent_step_gain_on_tie` |
| v5 不覆盖、谱系、profile 解锁、v4 兼容 | `test_generation_checkpoint_is_v5_immutable_and_records_lineage_and_evidence`、`test_v5_live_loader_enables_full_predictive_safe_mask_authority_without_changing_v4` |
| 活动指针原子切换、哈希检查、显式路径固定 | `test_active_checkpoint_pointer_is_atomic_hash_checked_and_explicit_path_is_pinned` |
| 无参数自训练与显式确定性入口 | `test_official_main_entry_defaults_to_v37_conservative_345_candidate`、`test_explicit_checkpoint_live_or_validate_only_never_starts_gradient_updates` |
| 新旧 CSV／SVG 跨点击追加 | `test_self_training_reports_append_across_clicks_and_render_all_requested_curves` 及旧日志测试 |

相关 8 个 Python 文件又通过 `py_compile`；`git diff --check` 无空白错误，仅报告 Windows
工作树既有的 LF→CRLF 提示。尝试对两个新模块运行 `pytest-cov` 时，本地环境没有安装
该插件，pytest 返回“unrecognized arguments”；没有为生成数字而临时安装依赖，也不
声称覆盖率百分比。

## 5. 全部项目 Markdown 审查

排除 `.codex`、`.venv`、`.pytest_cache` 和第三方依赖后，审查 11 份项目 Markdown：

| 文件 | 处理 |
|---|---|
| `README.md` | 更新无参数自训练、入口门禁、UI 字段、新日志与未验证边界 |
| `usvlib4ros_main_README.md` | 更新样例入口和 `--validate-only` |
| `海航赛智能船导航算法审查总结.md` | 更新当前链路、放行状态和自训练审查结论 |
| `PLAN (1).md` | 标记离线实施完成、Unity 待用户验证 |
| `reports/README.md` | 增加状态目录、2 个 CSV 和 4 个 SVG 的字段说明 |
| `reports/training_logging.tdd.md` | 保留旧日志证据并标注其“无梯度”结论已被后续实现取代 |
| `reports/planning_deferred_retry.tdd.md` | 历史证据仍正确，无需改写 |
| `reports/point4_point5_slow_turn.tdd.md` | 冻结的旧候选证据仍正确，无需改写 |
| `reports/points3_to5_conservative_reset.tdd.md` | 保守控制和已撤回 Reset 历史仍正确，无需改写 |
| `reports/zero_clearance_5000_steps.tdd.md` | 冻结的零净空／5000 步证据仍正确，无需改写 |
| `reports/self_training.tdd.md` | 本文，新增当前实现证据 |

本地相对链接检查结果：`PROJECT_MARKDOWN=11`、`BROKEN_LINKS=NONE`。

## 6. 未验证边界

- 没有启动、点击或操纵 Unity；5 个基线、5 个探索和 5 个独立验收仍待用户实际运行。
- 没有连接 ROS、Simulink 或 MATLAB，没有生成真实 v5 checkpoint 或活动指针。
- 没有运行全量 pytest，也没有重跑 13/13 零净空全路线回放。
- 聚焦测试证明编排和安全契约可以离线执行，不证明 Unity 船模能够完成 13 点或模型
  必然随轮次提升；严格门槛允许“不晋级”。
