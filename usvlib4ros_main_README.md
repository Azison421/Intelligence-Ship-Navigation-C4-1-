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
```

CLI 无参数运行默认选择 `unity_test` 和当前 v37 候选，供用户点击 Train 开展 Unity
复测。该候选只完成聚焦离线回放，尚未完成正式晋级。显式 `live` 仍只接受
`offline_ready=true`、`live_ready=true` 且 Unity 证据充分的 checkpoint；程序化
`USVNavMain.start(host, port, deviceId)` 的默认模式也仍为 `live`。

程序在连接前执行资产 preflight；正式任务按单回合运行，13 点完成后保持零控制；
退出时请求停止、发布零控制、等待导航线程并终止 ROS 连接。

每次 Train 按钮触发会分配独立 `run_id`；episode 结束后在项目 `reports/` 中累计
写入 3 份 CSV，并重绘总步长、逐目标累计步长和分段步长 3 张 SVG。报告写入失败
只记录错误，不会改变安全控制或样例 UI 协议。

不存在旧文档曾列出的 `--host`、`--device-id`、`enable_debug` 或各类 debug
frequency 参数。
