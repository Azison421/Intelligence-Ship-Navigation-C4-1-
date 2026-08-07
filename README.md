# 海航赛 2026 C4：National_Test 13 点导航

本仓库只支持固定 `National_Test` 地图、已知静态障碍和下方出生点。比赛入口
[`usvlib4ros/main.py`](./usvlib4ros/main.py) 保持字节不变；内部运行链已收敛为一个
规划无关的 V6 SAC 控制器。

## 当前边界

- 13 个原始航点必须按顺序进入 0.5 m 目标圆；冻结走廊中的引导点不计分。
- RRT* 只作为离线走廊生成工具，控制周期不进行在线规划或重规划。
- `NavigationStatus` 是唯一命令输出；`DeviceStatus` 只作为设备反馈和 ROS 时钟来源。
- 常规区域地图额外净空为 0.2 m、激光急停为 0.6 m。仅第 11 点至第 12 点的
  哈希绑定狭窄区按实船极限能力使用 0 m／0.4 m；离开该区立即恢复常规阈值，
  船体与障碍相交始终判为碰撞。
- 五个动作固定为经 Unity 标定的硬左、软左、直行、软右、硬右，百分比取整后必须
  互不重复。极左和极右之间必须先执行直行过渡。
- 连续 10 个新鲜周期没有安全动作时归零并截断；陈旧输入只安全停车，不记为策略
  失败。
- 已执行非零油门但连续约 3 s 低速且位移不足 0.05 m 时按 `MOTION_STALLED`
  截断；Unity 门控随后请求复位，不再等待 600 s。
- 不读取 v37 或更早的权重、Q 值、回放和训练状态。旧 checkpoint 与日志仅作历史
  证据保留。
- V3 观测先按物理量固定尺度归一化，再进入每个 GRU 头的 PyTorch `LayerNorm`；
  回放只保留最近 64 个完整 episode，避免量程主导门控和训练状态无界增长。

控制链：

```text
输入校验
  -> 航点推进
  -> local-waypoint-observation-v3
  -> 执行器可达动作
  -> 2 s 预测安全 mask
  -> 循环离散 SAC
  -> 最终安全复核
  -> NavigationStatus
```

## 安装

建议使用 Python 3.12。项目现有依赖已经足够，不需要额外规划或报告库；完整安装命令见
下方“完整运行顺序”的第 0 步。

连接参数只从本地 `config.json` 的 `ros2` 节读取，不应写入文档、checkpoint manifest
或日志。

## 完整运行顺序（先离线预训练，再真实训练）

下面按实际依赖顺序列出命令。`offline_validation`、`unity_test` 和 `live` 是不同
运行分支，不要把它们全部连续执行。

### 0. 一次性准备 Python 环境

```powershell
Set-Location 'G:\NavAlg - 副本-续训'
python -m pip install -r requirements.txt
```

### 1. 启动 ROS 和 Unity

先启动 `G:\智能导航C4-2026` 中的 ROS 环境，再打开 Unity：

1. 加载 `National_Test` 地图。
2. 确认 ROS IP、`DeviceStatus` 和 `NavigationStatus` 连接正常。
3. 点击 Unity“开始训练”。

这一步只是准备运行环境；下面的标定是一次短时真实执行器标定，不是策略训练。

### 2. 双向油门/舵角标定

保持 Unity 和 ROS 运行，在项目终端执行：

```powershell
python tools\calibrate_forward_control.py
```

成功后会生成类似下面的日志，实际时间戳以终端输出为准：

```text
artifacts\logs\forward-control-v2-20260805-173657.json
```

标定只向 `NavigationStatus` 发布短时有界命令，并等待船体在每次试验前重新静止。
日志 schema 为 `national-test-forward-calibration-v2`。

### 3. 离线 SAC 预训练

将上一步生成的实际标定日志传给训练入口：

```powershell
python tools\train_fixed_map_sac.py `
  --calibration-log "artifacts\logs\forward-control-v2-20260805-173657.json" `
  --seed 20260805
```

训练从随机 V6 网络开始；如果已有状态文件，命令会从上次进度继续。训练按 100 局
离线训练、20 局确定性评估的门控块进行，最多 1000 局；达到 20/20 后才生成 Unity
适应 checkpoint，失败则明确进入 `TRAINING_GATE_FAILED`。第一个 100 局训练块使用
冻结走廊的 1 m pure-pursuit 前视动作建立可学习回放，actor 使用掩码行为克隆预热，
双 critic 仍按 SAC 奖励更新。第 101 局起使用 DAgger：critic 只学习实际执行动作，
actor 用当前策略访问状态上的走廊专家标签抑制分布漂移；监督损失低于置信门后才逐步
放开 SAC actor 项。聚合阶段使用与预热相同的 Adam 学习率和每局最多 32 次更新；
第 101 局起奇数局采集确定性完整路线，偶数局保留随机分段或随机完整路线。评估和
Unity 运行均没有专家 fallback。随机
出生点若已落入当前目标圆会被重采样，保证每个训练局都有真实转移；离线单局上限与
Unity 一致为 600 s。

需要在累计 64 局处暂停并做 Unity 诊断时，追加
`--stop-after-training-episodes 64`。暂停后状态保持 `OFFLINE_TRAIN`；以后去掉该参数
会从第 65 局继续，而不是重新开始或额外训练 1000 局。

状态和证据位置：

- 进度：`artifacts\logs\national_test_training_progress.json`
- 可恢复训练状态：`artifacts\checkpoints\national_test_self_training_v6.pt`
- 每局离线证据：`artifacts\logs\national-test-offline-training-*.jsonl`

截至 2026-08-07，V6 在第 400 个训练回合后完成正式 20/20 离线评估，20 局均到达
全部 13 点，且无碰撞、超时或 `NO_SAFE_ACTION`。当前活动阶段是 `UNITY_ADAPT`，
checkpoint 为
`national_test_sac_v6_seed20260805_g4_t400_unity_adapt_a0_v0.pt`。这只解锁 Unity
适应，不等同于比赛模型已经晋级。

### 4. 导出离线训练报告

```powershell
python tools\export_training_reports.py
```

该命令从最新证据日志生成 CSV/SVG，不在训练或控制循环内绘图。续训跨越多个日志时，
可重复传入 `--offline-log`，并用
`--output-dir "reports\national_test_v6_stability"` 合并到独立目录。

### 5. 可选：生成离线确定性诊断 checkpoint

只有在训练状态仍为 `OFFLINE_TRAIN`、需要提前做诊断时才执行：

```powershell
python tools\prepare_unity_diagnostic.py
```

该命令不会注册 active checkpoint，也不替代正式离线门控。正式门通过后不需要执行。

### 6. 真实 Unity 适应与验收

离线门通过后，保持 Unity 和 ROS 环境运行，在项目根目录执行：

```powershell
python usvlib4ros\main.py
```

依次完成 5 局 Unity 适应和 5 局确定性验收：

1. 适应局更新权重，验收局冻结权重。
2. 每个计分回合结束后程序自动请求 Unity 复位，并等待复位状态确认；若复位后
   Unity 显示未开始，再点击一次“开始训练”进入下一局。
3. 操作者中止和输入陈旧只安全停车，不计入失败成绩。
4. 只有连续 5/5 完成 13 点、零碰撞、零超时、零 `NO_SAFE_ACTION`，才写出默认
   `artifacts\checkpoints\national_test_sac_checkpoint_v6.pt`。

### 7. 导出 Unity 运行报告

找到实际生成的运行日志，例如：

```text
artifacts\logs\national-test-runtime-20260806-120000.jsonl
```

再执行：

```powershell
python tools\export_training_reports.py `
  --runtime-only `
  --runtime-log "artifacts\logs\national-test-runtime-20260806-120000.jsonl" `
  --output-dir "reports\national_test_v6_unity_training"
```

该目录只保存 Unity 控制周期、episode 结果和适应 loss 的 CSV/SVG。

### 8. 已发布模型的正式运行分支

如果 active registry 已登记正式 checkpoint，直接运行即可：

```powershell
python usvlib4ros\main.py
```

需要明确指定正式模型时使用：

```powershell
python usvlib4ros\main.py `
  --policy-mode live `
  --checkpoint "artifacts\checkpoints\national_test_sac_checkpoint_v6.pt"
```

只做离线确定性检查时，先导出一个不登记为 active 的诊断 checkpoint：

```powershell
$diagnostic = python tools\prepare_unity_diagnostic.py | ConvertFrom-Json
python usvlib4ros\main.py `
  --policy-mode offline_validation `
  --checkpoint $diagnostic.checkpoint `
  --validate-only
```

只做明确的 Unity 确定性验收时使用：

```powershell
python usvlib4ros\main.py `
  --policy-mode unity_test `
  --checkpoint "artifacts\checkpoints\national_test_sac_checkpoint_v6.pt" `
  --validate-only
```

旧 checkpoint、旧日志和旧报表不迁移、不继续追加，也不参与当前模型加载。

## 关键资产

- 冻结走廊：`usvlib4ros/mapping/data/national_test_fixed_route_corridor_v1.json`
- 地图 sidecar：`usvlib4ros/mapping/data/beihu_static_world_sidecar.json`
- 现场仿射：`usvlib4ros/mapping/data/national_test_live_profile.json`
- V3 观测与 SAC：`usvlib4ros/policy/recurrent_sac.py`
- 唯一状态转换引擎：`usvlib4ros/navigation/fixed_map_runtime.py`
- ROS 生命周期与遥测：`usvlib4ros/navigation/fixed_map_service.py`

运行遥测逐周期记录策略动作、安全 mask、最终百分比命令和设备回读。历史 checkpoint
与现场日志不参与当前加载，但不删除，以便追溯。

## 验证规则

不运行全量回归。只执行与当前改动直接相关的精确测试节点；离线测试或合成回放不能
替代 Unity 5/5 验收。赛事背景资料见
[主办方主页](https://spaitlab.github.io/Maritime-Intelligent-Navigation-2026/)。
