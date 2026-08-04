# 海航赛 2026 C4 固定地图导航

本仓库保留比赛样例的 `main.py`、`USVNavMain.start(host, port, deviceId)`、
`DQN_NAV` 包装器以及 Unity 的 Reset／自动模式／路线／Train 触发流程，内部实现为
固定 `National_Test` 地图的 13 点导航。

> 2026-08-04 状态：13/13 的纯离线闭环用例已经通过；第 11 点入口已改为
> 不依赖 SAC 的确定性闭环轨迹跟踪，进入目标后再按原路倒车并绕障退出。
> 本次修改没有运行 Unity、ROS 或 MATLAB，也没有执行全量 pytest。
> 当前默认 checkpoint 仍是旧路线契约且 `offline_ready=false`、
> `live_ready=false`，因此正式 `live` 模式会按设计拒绝启动，不能把离线结果
> 表述为比赛环境已通过。

## 地图与任务约定

- Unity 图片中黄色圆点是 13 个必须按序到达的目标点。
- 白色线条是目标点连接关系，不等同于可直接跟踪的无碰撞轨迹。
- 红白相间物体是静态障碍物。
- 船从画面下方进入；画面上北、下南、左西、右东。
- 内部局部坐标为 `x=东`、`y=北`。
- 只有船体中心进入原始目标点的 0.5 m 圆域才切换下一点；规划辅助门不能替代
  正式目标。

比赛参考资料：

- [2026 海航赛智能导航主页](https://spaitlab.github.io/Maritime-Intelligent-Navigation-2026/)
- [赛事资源](https://spaitlab.github.io/Maritime-Intelligent-Navigation-2026/%E8%B5%9B%E4%BA%8B%E8%B5%84%E6%BA%90/)
- [操作手册](https://spaitlab.github.io/Maritime-Intelligent-Navigation-2026/%E6%93%8D%E4%BD%9C%E6%89%8B%E5%86%8C/)
- [评分细则](https://spaitlab.github.io/Maritime-Intelligent-Navigation-2026/%E8%AF%84%E5%88%86%E7%BB%86%E5%88%99/)

## 当前算法

控制链为：

```text
固定地图与 13 点任务
  -> Kinodynamic Informed RRT* 规划器实现
  -> SAC 五动作局部决策／局部确定性机动
  -> 预测安全监督器
  -> 样例 ROS/Unity 控制接口
```

“Informed RRT* + SAC”需要准确理解：普通航段使用
`KinodynamicInformedRRTStarPlanner` 与 SAC；运行时默认
`optimize_with_rrtstar=False`，规划器在首条经过动力学和碰撞验证的
grid/lattice seed 或连接轨迹上早停，以控制固定地图在线延迟。只有显式开启
`optimize_with_rrtstar=True` 才会继续执行 informed sampling 与 rewiring。
所以当前可运行配置不是“每段都用完整预算做 RRT* 最优化”。

局部点位允许使用不同算法，当前分工如下：

| 点位 | 控制策略 | 目的 |
|---|---|---|
| 1–3、7–10 | 动力学规划 + SAC + 安全动作掩码 | 普通航段跟踪和避障 |
| 4 | 确定性进点复合轨迹，持续到西侧安全交接门 | 避免进点后过早切换导致贴障 |
| 5 | 西侧交接门到第 5 点的固定转弯 | 保持连续可执行转向 |
| 6 | 局部动力学 primitive 搜索 | 穿过贴近浮标的有效目标区域并退出 |
| 11 | 确定性闭环入口 + 入点后反向退出 | 防止开环误差在窄区累积；入口不调用 SAC |
| 12 | 第 11 点后的确定性南侧绕障交接 | 先脱离窄区，再进入后续目标 |
| 13 | 确定性终点接近轨迹 | 保证最终进入 0.5 m 目标圆并停车 |

所有分支仍由预测安全监督器进行碰撞检查；没有安全动作时输出零控制并停止，
而不是强行执行局部机动。

路线指导契约是 `national-test-reversible-composite-v37`。本次版本升级用于阻止
旧 checkpoint 与当前局部点位进出逻辑静默混用。

## 环境与配置

建议使用 Python 3.10+。依赖以 `requirements.txt` 为准：

```powershell
Set-Location '<项目根目录>'
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

仓库根目录的 `config.json` 是连接参数唯一来源。不要把真实主机地址或设备 ID
复制到日志、截图、报告或提交中。结构如下：

```json
{
  "ros2": {
    "host": "<ROSBRIDGE_HOST>",
    "port": 9090,
    "deviceId": "<UNITY_DEVICE_ID>"
  }
}
```

运行前还需要按比赛平台手册启动 Unity、ROSbridge 与 Simulink。程序会在建立
外部连接前检查 checkpoint、manifest、静态地图和标定资产；缺失或契约不一致时
直接失败。

## 入口与运行模式

查看当前真实参数：

```powershell
.\.venv\Scripts\python.exe usvlib4ros\main.py --help
```

CLI 只支持：

```text
--config PATH
--policy-mode {live,offline_validation,unity_test}
--checkpoint PATH
```

不存在 `--host`、`--device-id` 或旧 README 中的调试频率参数；连接值来自
`config.json`。

模式含义：

| 模式 | checkpoint 门槛 | 用途 |
|---|---|---|
| `live` | `offline_ready=true`、`live_ready=true`，且具备足够 Unity 证据 | 正式比赛入口 |
| `offline_validation` | `offline_ready=true` | 离线验收工具 |
| `unity_test`（CLI 默认） | 可加载未晋级候选，但仍校验文件与算法契约 | Unity 候选测试 |

直接运行 `python usvlib4ros\main.py` 会选择当前 v37 候选用于 Unity 复测；该候选仅
完成聚焦离线回放，manifest 仍为 `offline_ready=false`、`live_ready=false`。正式运行
命令应在 v37 checkpoint 完成晋级后再使用：

```powershell
.\.venv\Scripts\python.exe usvlib4ros\main.py `
  --config .\config.json `
  --policy-mode live `
  --checkpoint .\artifacts\checkpoints\PROMOTED_V37_CHECKPOINT.pt
```

样例程序仍可调用：

```python
nav = USVNavMain.start(host, port, deviceId)
```

前三个位置参数及 UI 触发顺序保持兼容；新增的运行模式与 checkpoint 是仅限关键字
的可选参数。正式模式默认单回合：完成 13 点后持续发布零控制，不再自动 Reset
下一回合。`Ctrl+C`／`SIGTERM` 会请求停止、发布零控制、等待导航线程并关闭连接。

## 训练日志

每次在 Unity 中点击“开始训练”，程序会创建新的 `run_id`。每个 episode 结束后，
以下累计日志自动写入根目录 `reports/`：

- `training_runs.csv`：每次点击及其 episode／成功数；
- `training_episodes.csv`：每回合总步长和到达终点所需总步长；
- `waypoint_steps.csv`：13 个目标点的累计步长与相邻点分段步长；
- `training_total_steps.svg`：各回合总步长曲线；
- `waypoint_cumulative_steps.svg`：13 个目标点累计步长曲线；
- `waypoint_segment_steps.svg`：13 个航段步长曲线。

历史 CSV 不会因再次启动程序而清空；SVG 会在追加 episode 后用全部历史数据重绘。
未抵达目标点使用空值和曲线断点，不写成 0。图形为白底灰黑线、实线／虚线／点线
的论文风格，并且不依赖 Matplotlib。字段定义见
[`reports/README.md`](./reports/README.md)。

这里的 UI“训练”是 Unity 闭环运行，不会在线更新 SAC 权重；真正的离线 SAC 训练
仍由 `tools/train_fixed_map_sac.py` 执行，两者的步数含义不能混合。

## 离线验证边界

本次只使用点名用例，未执行 `pytest` 全量回归：

```powershell
# 13 个目标依次进入、点 11 退出后到点 12、最终完成且全程安全
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_fixed_map_trainer.py::test_fixed_map_sac_trains_complete_safe_episode_and_saves_checkpoint

# 点 11 闭环入口不调用 SAC；点 11 倒车出口不调用 SAC
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_fixed_map_runtime.py::test_runtime_executes_closed_loop_narrow_ingress_without_calling_sac `
  tests/test_fixed_map_runtime.py::test_runtime_executes_safe_reverse_escape_without_calling_sac

# 样例入口和界面契约
.\.venv\Scripts\python.exe -m pytest -q tests/test_sample_entry_compatibility.py

# Train 点击后的 CSV/SVG 日志
.\.venv\Scripts\python.exe -m pytest -q tests/test_training_reports.py
```

离线通过只证明当前简化动力学、静态 sidecar 和测试初始条件下可完成 13/13；它不
证明 Unity 中的实际船模、ROS 时序、传感器延迟或 Simulink 联调已经通过。

## 当前已知限制

- CLI 无参数入口选择已完成 13/13 聚焦离线回放的 v37 Unity 候选；它尚未晋级，
  因此显式 `live` 继续失败关闭是预期行为。
- 现有历史 Unity 候选日志含碰撞，且发生在本次修复之前；需要用户按相同 13 点
  地图重新测试，不能沿用旧日志给当前代码背书。
- `setup.py` 未声明运行依赖和地图 package data；当前只支持从源码目录运行，wheel
  交付尚未完成。
- `usvlib4ros（1）` 是仓库中的旧副本，未删除；正式入口只使用 `usvlib4ros/`。
- 传感器新鲜度仍部分依赖回调替换消息对象，若未来改成原地更新字段，需要改为
  接收时间戳或序列号。
- 当前范围是已知静态障碍的固定 National_Test 地图；动态障碍、随机地图和任意
  点数任务不在本次实现范围。

详细证据和逐点审查见
[`海航赛智能船导航算法审查总结.md`](./海航赛智能船导航算法审查总结.md)。
