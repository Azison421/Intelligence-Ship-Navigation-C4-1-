# 海航赛 2026 C4：National_Test 13 点导航

本仓库只支持固定 `National_Test` 地图、已知静态障碍和下方出生点。比赛入口
[`usvlib4ros/main.py`](./usvlib4ros/main.py) 保持字节不变；内部运行链已收敛为一个
规划无关的 V6 SAC 控制器。

## 当前边界

- 13 个原始航点必须按顺序进入 0.5 m 目标圆；冻结走廊中的引导点不计分。
- RRT* 只作为离线走廊生成工具，控制周期不进行在线规划或重规划。
- `NavigationStatus` 是唯一命令输出；`DeviceStatus` 只作为设备反馈和 ROS 时钟来源。
- 地图额外净空固定为 0.2 m，激光急停固定为 0.6 m，不允许 manifest 覆盖。
- 五个动作固定为经 Unity 标定的硬左、软左、直行、软右、硬右，百分比取整后必须
  互不重复。极左和极右之间必须先执行直行过渡。
- 连续 10 个新鲜周期没有安全动作时归零并截断；陈旧输入只安全停车，不记为策略
  失败。
- 不读取 v37 或更早的权重、Q 值、回放和训练状态。旧 checkpoint 与日志仅作历史
  证据保留。
- V3 原始观测进入每个 GRU 头前使用 PyTorch `LayerNorm`；回放只保留最近 64 个
  完整 episode，避免门控饱和和训练状态无界增长。

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

建议使用 Python 3.12。项目现有依赖已经足够，不需要额外规划或报告库。

```powershell
Set-Location 'G:\NavAlg - 副本'
python -m pip install -r requirements.txt
```

连接参数只从本地 `config.json` 的 `ros2` 节读取，不应写入文档、checkpoint manifest
或日志。

## 标定、离线训练和 Unity 门控

### 1. 双向控制标定

打开 Unity、加载 `National_Test` 并点击“开始训练”，然后运行：

```powershell
python tools\calibrate_forward_control.py
```

标定只向 `NavigationStatus` 发布短时有界命令，并等待船体在每次试验前重新静止。
成功日志 schema 为 `national-test-forward-calibration-v2`。

### 2. 离线 SAC 门控

将成功标定日志传给训练入口：

```powershell
python tools\train_fixed_map_sac.py `
  --calibration-log artifacts\logs\forward-control-v2-YYYYMMDD-HHMMSS.json
```

训练从随机 V6 网络开始。每轮严格执行 100 局训练和 20 局完整路线确定性评估；只有
20/20 才生成 Unity 适应 checkpoint。最多计 1000 个训练回合，仍未达标时状态变为
`TRAINING_GATE_FAILED`。进度写入
`artifacts/logs/national_test_training_progress.json`，完整可恢复状态写入
`artifacts/checkpoints/national_test_self_training_v2.pt`；每局证据同时追加到本次运行的
`national-test-offline-training-*.jsonl`。

需要 CSV/SVG 时从证据日志生成，不在训练或控制循环内绘图：

```powershell
python tools\export_training_reports.py
```

Unity 运行后可额外传入
`--runtime-log artifacts\logs\national-test-runtime-YYYYMMDD-HHMMSS.jsonl`
的具体文件，生成 V6 控制周期 CSV 与 SVG。旧版报表不迁移、不继续追加。

### 3. Unity 适应与验收

离线门通过后，保持 Unity 与 ROS 环境运行：

```powershell
python usvlib4ros\main.py
```

依次完成 5 局 Unity 适应和 5 局确定性验收。适应局才更新权重；验收 5 局期间权重
冻结。每局结束后点击“结束训练”，等待船静止，再点击“开始训练”进入下一局。
操作员中止和输入陈旧不计成绩。只有连续 5/5 完成 13 点、零碰撞、零超时、零
`NO_SAFE_ACTION`，才写出默认 `national_test_sac_checkpoint_v6.pt`。

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
