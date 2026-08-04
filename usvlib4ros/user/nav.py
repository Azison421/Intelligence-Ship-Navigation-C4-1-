"""Official sample entry name backed by the fixed National_Test algorithm."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from usvlib4ros.navigation.device_action_bridge import (
    create_ros_device_action_bridge,
)
from usvlib4ros.navigation.fixed_map_service import (
    FixedMapNavigationService,
)
from usvlib4ros.policy.checkpoint_promotion import PolicyMode


class DQN_NAV:
    """Compatibility wrapper retained for ``usvlib4ros.main``.

    The official entry runs in :attr:`PolicyMode.LIVE` by default, which
    requires a checkpoint that passed both offline and Unity live
    validation.  Candidate loaders are reachable only through an explicit
    validation mode.
    """

    Instance = None

    def __init__(
        self,
        ros_ctrl,
        global_data,
        xyzAxis=True,
        *,
        policy_mode: PolicyMode = PolicyMode.LIVE,
        checkpoint_path: Optional[Path] = None,
        single_episode: bool = True,
        self_training: bool = False,
        validate_only: bool = False,
    ):
        del xyzAxis
        self.ros_ctrl = ros_ctrl
        self.global_data = global_data
        self.policy_mode = PolicyMode(policy_mode)
        self.navThread = None
        self._stop_event = threading.Event()
        device_id = getattr(ros_ctrl, "deviceId", None)
        self._action_bridge = (
            create_ros_device_action_bridge(device_id)
            if device_id
            else None
        )
        self._service = FixedMapNavigationService(
            ros_ctrl,
            global_data,
            action_bridge=self._action_bridge,
            policy_mode=self.policy_mode,
            checkpoint_path=checkpoint_path,
            single_episode=single_episode,
            self_training=self_training,
            validate_only=validate_only,
        )

    def startService(self):
        if self._action_bridge is not None:
            self._action_bridge.start(publish_hz=30.0)
        self.navThread = threading.Thread(target=self.run)
        self.navThread.start()

    def run(self):
        self._service.run()

    def is_running(self) -> bool:
        return (
            self.navThread is not None
            and self.navThread.is_alive()
        )

    def stop(self):
        """Request a clean shutdown: stop, zero control, join the thread."""

        self._stop_event.set()
        self._service.request_stop()
        self._publish_zero()
        if self._action_bridge is not None:
            self._action_bridge.close()
        thread = self.navThread
        if (
            thread is not None
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=5.0)

    def _publish_zero(self):
        self.global_data.updateThrottleRudderOutput(
            0,
            0,
            0.0,
            0,
            0.0,
        )
        if self._action_bridge is not None:
            self._action_bridge.set_command(0, 0)


__all__ = ["DQN_NAV"]
