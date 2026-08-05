"""Entry adapter required by the immutable official ``main.py``."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from usvlib4ros.navigation.fixed_map_service import (
    FixedMapNavigationService,
)
from usvlib4ros.policy.checkpoint_promotion import PolicyMode


class DQN_NAV:
    """Bind the immutable official entry to the National_Test service."""

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
        self._service = FixedMapNavigationService(
            ros_ctrl,
            global_data,
            policy_mode=self.policy_mode,
            checkpoint_path=checkpoint_path,
            single_episode=single_episode,
            self_training=self_training,
            validate_only=validate_only,
        )

    def startService(self):
        self.navThread = threading.Thread(
            target=self.run,
            name="national-test-navigation",
        )
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

        self._service.request_stop()
        self._publish_zero()
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


__all__ = ["DQN_NAV"]
