"""Repository boundary checks for the National_Test SAC refactor."""

from __future__ import annotations

import hashlib
from pathlib import Path

import setuptools


ROOT = Path(__file__).resolve().parents[1]
MAIN_SHA256 = "dcab3c5f60d1357866015e77073f2ad403bf1e3aee1a4fca7be319d39996b192"


def test_main_entry_matches_approved_hash():
    main_path = ROOT / "usvlib4ros" / "main.py"
    assert hashlib.sha256(main_path.read_bytes()).hexdigest() == MAIN_SHA256


def test_setup_packages_only_current_usvlib4ros_tree():
    packages = set(
        setuptools.find_packages(
            where=str(ROOT),
            include=["usvlib4ros", "usvlib4ros.*"],
        )
    )
    assert packages
    assert all(name == "usvlib4ros" or name.startswith("usvlib4ros.") for name in packages)
    assert not any("（1）" in name for name in packages)


def test_retired_source_paths_are_absent():
    retired = (
        "usvlib4ros（1）",
        "usvlib4ros/PPO_ship_obstacle_0.pth",
        "usvlib4ros/user/PP0.py",
        "usvlib4ros/user/PP0_1.py",
        "usvlib4ros/user/nav_1.py",
        "usvlib4ros/navigation/device_action_bridge.py",
        "usvlib4ros/navigation/auto_navigation_example.py",
        "usvlib4ros/navigation/autp_pilot_service.py",
        "usvlib4ros/navigation/avoid_collision_service.py",
        "usvlib4ros/navigation/route_plan_service.py",
        "usvlib4ros/vehicle",
        "usvlib4ros/usvRosUtil/ros_bag_util.py",
        "usvlib4ros/usvRosUtil/nmea_util.py",
        "usvlib4ros/usvRosUtil/usv_util.py",
        "usvlib4ros/msg/parameter.py",
    )
    assert [path for path in retired if (ROOT / path).exists()] == []
