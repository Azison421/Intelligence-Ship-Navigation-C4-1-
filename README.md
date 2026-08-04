# 海航赛 2026 C4 固定地图导航

本仓库保留比赛样例的 `main.py`、`USVNavMain.start(host, port, deviceId)`、
`DQN_NAV` 包装器以及 Unity 的 Reset／自动模式／路线／Train 触发流程，内部实现为
固定 `National_Test` 地图的 13 点导航。

> 2026-08-04 状态：无参数 CLI 已接入可中断、可续训的 SAC 混合自训练，以当前
> v37 零净空保守候选为初始冠军。每代执行 95 个离线训练回合、5 个 Unity 安全探索
> 回合、20 个离线评估回合和 5 个 Unity 确定性验收回合；首次训练前另做 5 个原始
> 冠军 Unity 基线回合。零净空、点 3→4 的 `0.1/±0.1` 硬上限、点 4→5 的
> `0.1/0.12` 硬上限、0.4 m 船体和预测安全否决权均保留。
> 自训练闭环已通过聚焦纯 Python 测试，但本次没有启动 Unity、ROS 或 MATLAB，也没有
> 运行全量 pytest；因此尚无 v5 晋级模型，初始 v37 候选仍为
> `offline_ready=false`、`live_ready=false`。

## 地图与任务约定

- Unity 图片中黄色圆点是 13 个必须按序到达的目标点。
- 白色线条是目标点连接关系，不等同于可直接跟踪的无碰撞轨迹。
- 红白相间物体是静态障碍物。
- 船从画面下方进入；画面上北、下南、左西、右东。
- 内部局部坐标为 `x=东`、`y=北`。
- 只有船体中心进入原始目标点的 0.5 m 圆域才切换下一点；规划辅助门不能替代
  正式目标。
- 旧 v37 基线的地图额外净空／激光提前急停阈值分别为 0.2 m／0.6 m；当前零净空
  Unity 实验分别为 0 m／0 m。船体外形、13 点顺序和主算法不变；当前默认候选只在
  点 3 后至点 5 的局部复合机动中使用低速 profile。

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
| 4 | 确定性进点复合轨迹；当前候选从点 3 后把正向油门和舵角绝对值都限制为 0.1，持续到西侧安全交接门 | 低速、小舵角通过目标并避免过早切换贴障 |
| 5 | 西侧交接门以 `0.1` 油门、`0.12` 舵角固定转弯进入 | 使用离线可到达点 5 的最小已验证舵角，缩小外飘 |
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
--validate-only
```

不存在 `--host`、`--device-id` 或旧 README 中的调试频率参数；连接值来自
`config.json`。

模式含义：

| 模式 | checkpoint 门槛 | 用途 |
|---|---|---|
| `live` | `offline_ready=true`、`live_ready=true`，且具备足够 Unity 证据 | 正式比赛入口 |
| `offline_validation` | `offline_ready=true` | 离线验收工具 |
| `unity_test`（CLI 默认模式） | 可加载未晋级候选，但仍校验文件与算法契约 | 无参数时混合自训练；显式模式时确定性候选测试 |

直接运行 `python usvlib4ros\main.py` 会选择当前活动冠军；首次没有活动指针时从
`national_test_sac_v37_zero_clearance_conservative_345_unity_test.pt` 热启动，并在
Unity 点击“开始训练”后进入混合自训练。该种子与旧
`national_test_sac_v37_unity_test.pt`、上一版零净空候选及上一版慢转候选的权重
字节和 SHA-256 完全相同，但由独立 manifest 绑定 0 m／0 m 安全参数，以及点 3→4
的 `0.1/±0.1` 包络和第 4→5 点 `Control(0.1, 0.12)`。三个旧路径仍使用各自原
配置。显式 `--checkpoint PATH`、显式 `--policy-mode` 或 `--validate-only` 都只做
确定性运行，不执行梯度更新；显式路径不会跟随活动冠军。四份 v37 候选的 readiness
都仍为 false。正式运行命令只能在兼容 checkpoint 完成自动或人工晋级后使用：

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

前三个位置参数及 UI 触发顺序保持兼容；新增的运行模式、checkpoint、自训练和
只验收标志都是仅限关键字的可选参数。单回合默认上限为 5000 个控制步、墙钟上限
为 600 s；旧 `MAX_EPOCH` 保持不变，自训练 UI 的 `MaxE` 显示当前累计目标。
`PLANNING_DEFERRED` 会在当前周期发布零控制并保留回合，下一周期继续尝试规划；不会
再因为连续出现 3 次而结束回合。既有 600 s 墙钟超时、碰撞和停止请求仍可结束回合。
`Ctrl+C`／`SIGTERM` 会请求停止、发布零控制、等待导航线程并关闭连接。

## 训练日志

每次在 Unity 中点击“开始训练”，程序会创建新的 `run_id`。每个 episode 结束后，
以下累计日志自动写入根目录 `reports/`：

- `training_runs.csv`：每次点击及其 episode／成功数；
- `training_episodes.csv`：每回合总步长和到达终点所需总步长；
- `waypoint_steps.csv`：13 个目标点的累计步长与相邻点分段步长；
- `training_total_steps.svg`：各回合总步长曲线；
- `waypoint_cumulative_steps.svg`：13 个目标点累计步长曲线；
- `waypoint_segment_steps.svg`：13 个航段步长曲线。

无参数自训练还会追加：

- `self_training_episodes.csv`：阶段、代次、reward、成功、步数、loss 和训练步；
- `self_training_generations.csv`：父／候选哈希、20/20 与 5/5 结果和晋级原因；
- `self_training_reward.svg`、`self_training_success_rate.svg`、
  `self_training_total_steps.svg`、`self_training_losses.svg`；
- `self_training_state/`：每个完整回合后的原子续训状态；半回合不会写入。

历史 CSV 不会因再次启动程序而清空；SVG 会在追加 episode 后用全部历史数据重绘，
坐标轴按实际数据自动缩放，5000 不是硬编码的图表上限。
未抵达目标点使用空值和曲线断点，不写成 0。图形为白底灰黑线、实线／虚线／点线
的论文风格，并且不依赖 Matplotlib。字段定义见
[`reports/README.md`](./reports/README.md)。

无参数入口中的 UI“训练”现在会更新 SAC：每个离线训练回合做 16 次更新，5 个
Unity 探索回合后做 80 次 6:2 混合更新。UI 的 `E` 是累计训练回合，`Step` 是
当前回合步数，`Score` 是累计 reward，`Loss` 是最新 critic loss，`MaxE` 是当前
累计目标。显式 checkpoint、显式模式或 `--validate-only` 仍是确定性运行且不更新。

## 离线验证边界

原 v37 基线的 13/13 证据继续保留，但零净空和第 3→4→5 点保守增量只使用点名用例，
未重新运行 13/13 回放，也未执行 `pytest` 全量回归。自训练闭环的状态机、回放、
权重更新、晋级和日志证据见
[`reports/self_training.tdd.md`](./reports/self_training.tdd.md)。零净空／5000 步证据见
[`reports/zero_clearance_5000_steps.tdd.md`](./reports/zero_clearance_5000_steps.tdd.md)，
本次局部机动证据见
[`reports/points3_to5_conservative_reset.tdd.md`](./reports/points3_to5_conservative_reset.tdd.md)；
上一版慢转证据仍保留在
[`reports/point4_point5_slow_turn.tdd.md`](./reports/point4_point5_slow_turn.tdd.md)。
覆盖重点为：

- 旧／新安全 profile、地图与路线 hash；
- 0–0.2 m 额外净空区间、船体接触和激光接触；
- 新旧候选加载门禁、5000 步／600 s；
- CSV 的 5000 步记录和 SVG 自动缩放。
- 点 3→4 的油门／舵角上限、点 4→5 最小可行转弯、规划延迟零控制重试、旧候选回退与新候选门禁。

原 v37 基线的离线通过只证明该基线在简化动力学、静态 sidecar 和测试初始条件下
可完成 13/13；它不证明新零净空配置或 Unity 中的实际船模、ROS 时序、传感器延迟、
Simulink 联调已经通过。

## 当前已知限制

- CLI 无参数入口会从未晋级的零净空保守 3→4→5 候选启动自训练；训练编排仅通过
  聚焦离线契约测试，尚无真实 Unity 自训练结果或 v5 晋级模型。显式 `live` 在没有
  活动冠军时失败关闭是预期行为。
- 用户截图显示旧候选在第 4→5 点存在碰撞，并出现持续规划延迟；当前保守 profile
  与规划延迟重试是针对这些现象的未验证修复，需要用户在相同 13 点地图重新测试。
- `setup.py` 未声明运行依赖和地图 package data；当前只支持从源码目录运行，wheel
  交付尚未完成。
- `usvlib4ros（1）` 是仓库中的旧副本，未删除；正式入口只使用 `usvlib4ros/`。
- 传感器新鲜度仍部分依赖回调替换消息对象，若未来改成原地更新字段，需要改为
  接收时间戳或序列号。
- 当前范围是已知静态障碍的固定 National_Test 地图；动态障碍、随机地图和任意
  点数任务不在本次实现范围。

详细证据和逐点审查见
[`海航赛智能船导航算法审查总结.md`](./海航赛智能船导航算法审查总结.md)。
