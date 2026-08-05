"""Policy modes required by the immutable competition entrypoint."""

from __future__ import annotations

from enum import Enum
class PolicyMode(str, Enum):
    """Checkpoint admission stage selected by ``main.py``."""

    LIVE = "live"
    OFFLINE_VALIDATION = "offline_validation"
    UNITY_TEST = "unity_test"
__all__ = ["PolicyMode"]
