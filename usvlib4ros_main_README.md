# `usvlib4ros/main.py` 不可变入口说明

`usvlib4ros/main.py` 的批准 SHA-256 为：

```text
dcab3c5f60d1357866015e77073f2ad403bf1e3aee1a4fca7be319d39996b192
```

生产改造不得修改该文件。保留 `DQN_NAV` 仅因为入口按名称导入它；该包装器直接绑定
当前 `FixedMapNavigationService`，不包含旧 DQN/PPO 或自动驾驶 fallback。

## 操作方式

离线门生成活动 V6 checkpoint 后：

```powershell
python usvlib4ros\main.py
```

入口先校验活动 registry、checkpoint/manifest 哈希、完整门控证据、冻结走廊和地图
资产，再建立 ROS 连接。registry 缺失、阶段不允许、晋级证据不足或 schema 不是 V6
时直接失败，不加载旧模型。

无参数运行会继续当前 Unity 门：先完成 5 局适应，再完成 5 局冻结验收。显式
`--validate-only` 不更新权重。完整 CLI 仍由不可变入口自身的 `--help` 输出为准。

## 输出边界

- 只向 `/usv/navigation/status/<deviceId>` 发布 `NavigationStatus`。
- `/usv/device/status/<deviceId>` 只订阅，不自发布、不自回读。
- 命令 timestamp 使用只读设备时钟，以满足 ROS 管理节点 2 s 新鲜度门。
- 任务结束、输入陈旧、安全截断、异常和进程退出均持续发布零油门/零舵角。
- 非零油门下约 3 s 无有效运动会以 `MOTION_STALLED` 截断，避免撞墙后卡到回合超时。
- Unity 门控中的计分回合结束后调用现有 Reset 服务；只有收到复位完成状态才进入
  下一回合等待。

连接参数来自本地 `config.json`，本文不记录具体地址和设备标识。完整架构与训练步骤
见根目录 [`README.md`](./README.md)。
