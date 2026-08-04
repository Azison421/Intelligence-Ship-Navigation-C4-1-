# `usvlib4ros/main.py` 入口兼容说明

完整运行说明、算法边界和离线证据以根目录
[`README.md`](./README.md) 为准。本文件只记录样例入口契约，避免两份运行手册继续
漂移。

## 保持不变的接口

```python
nav = USVNavMain.start(host, port, deviceId)
```

- `host`、`port`、`deviceId` 三个位置参数保持不变。
- `DQN_NAV` 类名和样例 UI 的 Reset／自动模式／路线／Train 流程保持不变。
- 连接值从 `config.json` 的 `ros2` 节读取；本文档不记录真实主机或设备标识。

## 新增但兼容的安全参数

```python
nav = USVNavMain.start(
    host,
    port,
    deviceId,
    policy_mode=PolicyMode.LIVE,
    checkpoint_path=checkpoint_path,
)
```

新增参数均为仅限关键字参数，不影响原三参数调用。CLI 为：

```text
python usvlib4ros/main.py
  [--config PATH]
  [--policy-mode {live,offline_validation,unity_test}]
  [--checkpoint PATH]
  [--validate-only]
```

CLI 无参数运行默认选择 `unity_test`，并在用户点击 Train 后启动可中断、可续训的
混合自训练。它优先使用 `national_test_sac_active.json` 指向的已晋级冠军；首次没有
活动指针时从 `national_test_sac_v37_zero_clearance_conservative_345_unity_test.pt`
热启动。该种子复用旧 v37 的同一份权重，manifest 继续绑定 0 m／0 m，并把
第 3→4 段限制为 `油门≤0.1、|舵角|≤0.1`，第 4→5 点转弯设为
`Control(0.1, 0.12)`；0.4 m 船体外形和预测碰撞仍保留。旧 v37、上一版零净空及
上一版慢转显式路径继续使用各自原局部控制。种子仅允许 `unity_test`，
`offline_ready`、`live_ready` 均为 false；只有满足 20/20 离线和 5/5 Unity 双门槛
并通过冠军比较的 v5 才会原子切换活动指针。显式 checkpoint、显式模式或
`--validate-only` 不启动梯度更新。程序化 `USVNavMain.start(host, port, deviceId)`
的默认模式仍为 `live`，`self_training` 默认仍为 false。

程序在连接前执行资产 preflight；单回合默认最多 5000 个控制步或 600 s，13 点完成
后保持零控制；退出时请求停止、发布零控制、等待导航线程并终止 ROS 连接。
`PLANNING_DEFERRED` 会发布零控制并在同一回合继续重试，不再按连续次数结束回合；
既有 600 s 墙钟超时、碰撞和停止请求仍保持原来的结束行为。

每次 Train 按钮触发仍写旧 3 份 CSV 和 3 张步长 SVG。无参数自训练另外写入
`self_training_episodes.csv`、`self_training_generations.csv` 以及 reward、累计成功率、
总步数和 actor/critic loss 4 张 SVG，并在每个完整回合后原子保存续训状态。
SVG 坐标轴按实际数据自动缩放，5000 不是图表硬上限。UI 字段继续复用原协议：
`E/Step/Score/Loss/MaxE` 分别表示累计训练回合、当前步、累计 reward、最新 critic
loss 和当前累计目标。

不存在旧文档曾列出的 `--host`、`--device-id`、`enable_debug` 或各类 debug
frequency 参数。
