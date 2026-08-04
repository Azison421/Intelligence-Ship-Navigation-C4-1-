"""Official Maritime Intelligent Navigation 2026 C4 entrypoint.

The no-argument sample entry loads the current focused-offline-replay
candidate in :attr:`PolicyMode.UNITY_TEST` so an operator can run Unity
validation.
Explicit :attr:`PolicyMode.LIVE` still requires full offline and Unity
promotion before a checkpoint is loaded.

CLI:
    --config PATH       path to config.json (default: <repo>/config.json)
    --policy-mode MODE  live | offline_validation | unity_test
    --checkpoint PATH   override the default SAC checkpoint path
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

# Allow ``python usvlib4ros/main.py`` to import the parent package.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

from usvlib4ros.policy.checkpoint_promotion import PolicyMode


def _default_config_paths() -> list[Path]:
    script_dir = Path(__file__).resolve().parent
    return [
        script_dir / "config.json",
        script_dir.parent / "config.json",
    ]


def load_config(config_path: Optional[str] = None) -> dict:
    """Load config.json; ``--config`` wins, otherwise well-known paths."""

    candidates = (
        [Path(config_path)]
        if config_path
        else _default_config_paths()
    )
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        "配置文件未找到，已尝试路径: "
        + ", ".join(str(path) for path in candidates)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Maritime Intelligent Navigation 2026 C4 entrypoint"
        )
    )
    parser.add_argument(
        "--config",
        default=None,
        help="path to config.json",
    )
    parser.add_argument(
        "--policy-mode",
        choices=[mode.value for mode in PolicyMode],
        default=PolicyMode.UNITY_TEST.value,
        help=(
            "model promotion gate; live requires a checkpoint that "
            "passed offline and Unity validation (default: unity_test)"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="override the default SAC checkpoint path",
    )
    return parser


def resolve_checkpoint_path(
    policy_mode: PolicyMode,
    checkpoint_path: Optional[str | Path],
) -> Path:
    """Select the CLI checkpoint without weakening explicit live gates."""

    if checkpoint_path is not None:
        return Path(checkpoint_path)
    from usvlib4ros.navigation.fixed_map_runtime import (
        DEFAULT_CHECKPOINT,
        DEFAULT_UNITY_TEST_CHECKPOINT,
    )

    if policy_mode == PolicyMode.UNITY_TEST:
        return DEFAULT_UNITY_TEST_CHECKPOINT
    return DEFAULT_CHECKPOINT


def preflight_assets(checkpoint_path: Optional[Path] = None) -> None:
    """Fail fast when checkpoint, manifest, or map assets are missing."""

    from usvlib4ros.navigation.fixed_map_runtime import DEFAULT_CHECKPOINT
    from usvlib4ros.navigation.fixed_map_service import (
        preflight_assets as _preflight,
    )

    _preflight(Path(checkpoint_path or DEFAULT_CHECKPOINT))


class USVNavMain:

    @classmethod
    def start(
        cls,
        host,
        port,
        deviceId,
        *,
        policy_mode: PolicyMode = PolicyMode.LIVE,
        checkpoint_path: Optional[Path] = None,
    ):
        # Heavy ROS-facing imports are deferred so that the module stays
        # importable (and testable) without a ROS bridge installation.
        from usvlib4ros import GlobalData
        from usvlib4ros.navigation.usv_ros2_controller import Ros2Controller
        from usvlib4ros.user.nav import DQN_NAV

        globalData = GlobalData().getInstance()
        rosCtrl = Ros2Controller(
            host=host,
            port=port,
            deviceId=deviceId,
            globalData=globalData,
        )
        nav = DQN_NAV(
            rosCtrl,
            globalData,
            policy_mode=policy_mode,
            checkpoint_path=checkpoint_path,
        )
        nav.startService()
        return nav


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    policy_mode = PolicyMode(args.policy_mode)
    checkpoint_path = resolve_checkpoint_path(
        policy_mode,
        args.checkpoint,
    )
    config = load_config(args.config)
    ros2_cfg = config["ros2"]

    preflight_assets(checkpoint_path)

    nav = USVNavMain.start(
        ros2_cfg["host"],
        ros2_cfg["port"],
        ros2_cfg["deviceId"],
        policy_mode=policy_mode,
        checkpoint_path=checkpoint_path,
    )
    print(
        f"policy_mode={policy_mode.value} "
        f"checkpoint={nav._service.checkpoint_path} "
        "waiting for the train trigger..."
    )

    stop_requested = threading.Event()

    def _request_stop(_signum, _frame):
        stop_requested.set()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    try:
        while not stop_requested.is_set():
            time.sleep(0.2)
    finally:
        nav.stop()
        from usvlib4ros.usvRosUtil import USVRosbridgeClient

        ros = USVRosbridgeClient.ros
        if ros is not None:
            ros.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
