import threading
from pathlib import Path
from types import SimpleNamespace

from usvlib4ros.navigation.fixed_map_runtime import (
    RuntimeDecision,
    build_live_route_context,
)
from usvlib4ros.navigation.fixed_map_service import (
    FAILURE_CONFIRMATION_SECONDS,
    FixedMapNavigationService,
    advance_collision_confirmation,
    advance_failure_streak,
    is_collision_evidence,
    policy_loader_for_mode,
)
from usvlib4ros.planning import Control
from usvlib4ros.policy.checkpoint_promotion import PolicyMode
from usvlib4ros.user.nav import DQN_NAV


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _OutputCapture:
    def __init__(self):
        self.throttle = None
        self.throttle_updates = 0
        self.algorithm = None
        self.device_data = SimpleNamespace()
        self.scada_data = SimpleNamespace()

    def updateThrottleRudderOutput(self, *values):
        self.throttle = values
        self.throttle_updates += 1

    def updateAlgorithmOutput(self, *values):
        self.algorithm = values


class _ActionBridgeCapture:
    def __init__(self):
        self.command = None

    def set_command(self, throttle, rudder):
        self.command = (throttle, rudder)


def test_confirmed_failure_streak_ignores_transient_invalid_samples():
    streak = 0
    for _ in range(9):
        streak = advance_failure_streak(
            streak,
            "MAP_INVALID",
            {"MAP_INVALID", "LASER_EMERGENCY_STOP"},
        )
    assert streak == 9
    assert streak < 10

    assert (
        advance_failure_streak(
            streak,
            "MAP_INVALID",
            {"MAP_INVALID", "LASER_EMERGENCY_STOP"},
        )
        == 10
    )
    assert (
        advance_failure_streak(
            streak,
            "POLICY_ACTION_SAFE",
            {"MAP_INVALID", "LASER_EMERGENCY_STOP"},
        )
        == 0
    )
    assert not is_collision_evidence("MAP_INVALID", 0.61)
    assert is_collision_evidence("MAP_INVALID", 0.6)
    assert not is_collision_evidence("DYNAMICS_INVALID", 0.5)
    assert not is_collision_evidence("LASER_EMERGENCY_STOP", 0.5)
    assert not is_collision_evidence("NO_SAFE_ACTION", 0.5)


def test_collision_requires_direct_dual_evidence_or_five_seconds_of_laser():
    assert FAILURE_CONFIRMATION_SECONDS == 5.0
    started, confirmed = advance_collision_confirmation(
        None,
        "LASER_EMERGENCY_STOP",
        0.5,
        now_s=100.0,
    )
    assert started == 100.0
    assert not confirmed

    started, confirmed = advance_collision_confirmation(
        started,
        "LASER_EMERGENCY_STOP",
        0.5,
        now_s=104.999,
    )
    assert not confirmed

    started, confirmed = advance_collision_confirmation(
        started,
        "LASER_EMERGENCY_STOP",
        0.5,
        now_s=105.0,
    )
    assert confirmed

    started, confirmed = advance_collision_confirmation(
        started,
        "NO_SAFE_ACTION",
        0.5,
        now_s=106.0,
    )
    assert started is None
    assert not confirmed

    started, confirmed = advance_collision_confirmation(
        None,
        "MAP_INVALID",
        0.6,
        now_s=200.0,
    )
    assert started is None
    assert confirmed


def test_official_main_entry_defaults_to_v37_unity_test_candidate():
    from usvlib4ros import main as entry

    args = entry.build_parser().parse_args([])

    assert args.policy_mode == PolicyMode.UNITY_TEST.value
    assert args.config is None
    assert args.checkpoint is None
    assert (
        entry.resolve_checkpoint_path(
            PolicyMode(args.policy_mode),
            args.checkpoint,
        ).name
        == "national_test_sac_v37_unity_test.pt"
    )


def test_official_main_entry_keeps_live_checkpoint_for_explicit_live_mode():
    from usvlib4ros import main as entry

    assert (
        entry.resolve_checkpoint_path(PolicyMode.LIVE, None).name
        == "national_test_sac_live_v10_tested.pt"
    )


def test_official_main_entry_accepts_explicit_policy_mode():
    from usvlib4ros import main as entry

    args = entry.build_parser().parse_args(
        [
            "--config",
            "cfg.json",
            "--policy-mode",
            "unity_test",
            "--checkpoint",
            "candidate.pt",
        ]
    )

    assert args.policy_mode == "unity_test"
    assert args.config == "cfg.json"
    assert args.checkpoint == "candidate.pt"


def test_policy_loader_maps_explicit_modes_to_fail_closed_loaders():
    from usvlib4ros.navigation.fixed_map_runtime import (
        load_live_ready_policy,
        load_offline_ready_policy,
        load_tested_candidate_policy,
    )

    assert policy_loader_for_mode(PolicyMode.LIVE) is load_live_ready_policy
    assert (
        policy_loader_for_mode(PolicyMode.OFFLINE_VALIDATION)
        is load_offline_ready_policy
    )
    assert (
        policy_loader_for_mode(PolicyMode.UNITY_TEST)
        is load_tested_candidate_policy
    )


def test_dqn_nav_defaults_to_live_policy_mode_and_starts_thread():
    navigation = DQN_NAV(object(), _OutputCapture())
    assert navigation._service.policy_mode == PolicyMode.LIVE
    assert navigation._service.single_episode
    called = threading.Event()
    navigation._service.run = called.set

    navigation.startService()
    navigation.navThread.join(timeout=1.0)

    assert called.is_set()


def test_dqn_nav_wires_and_starts_live_device_bridge(monkeypatch):
    action_bridge = SimpleNamespace(
        started=False,
        start=lambda **kwargs: setattr(
            action_bridge,
            "started",
            kwargs,
        ),
    )
    monkeypatch.setattr(
        "usvlib4ros.user.nav.create_ros_device_action_bridge",
        lambda device_id: action_bridge,
    )
    navigation = DQN_NAV(
        SimpleNamespace(deviceId="test-device"),
        _OutputCapture(),
    )
    called = threading.Event()
    navigation._service.run = called.set

    navigation.startService()
    navigation.navThread.join(timeout=1.0)

    assert navigation._service.action_bridge is action_bridge
    assert action_bridge.started == {"publish_hz": 30.0}
    assert called.is_set()


def test_sample_output_fields_receive_bounded_runtime_decision():
    output = _OutputCapture()
    action_bridge = _ActionBridgeCapture()
    service = FixedMapNavigationService(
        object(),
        output,
        action_bridge=action_bridge,
    )
    decision = RuntimeDecision(
        reason="POLICY_ACTION_SAFE",
        control=Control(throttle=0.25, rudder=-1.0),
        action=4,
        mission_index=3,
        distance_to_goal_m=4.5,
        advised_heading_deg=-30.0,
        safe_mask=(True,) * 5,
        completed=False,
        replanned=False,
    )

    service._publish_decision(decision, episode=2, step=7)

    assert output.throttle == (25, -100, -30.0, 3, 4.5)
    assert output.algorithm == (2, 7, 25, 0.0, 4_000, 2)
    assert action_bridge.command == (25, -100)


def test_sample_output_preserves_reverse_throttle_end_to_end():
    output = _OutputCapture()
    action_bridge = _ActionBridgeCapture()
    service = FixedMapNavigationService(
        object(),
        output,
        action_bridge=action_bridge,
    )
    decision = RuntimeDecision(
        reason="OVERSPEED_REVERSE_BRAKE",
        control=Control(throttle=-0.4, rudder=0.2),
        action=2,
        mission_index=10,
        distance_to_goal_m=0.3,
        advised_heading_deg=90.0,
        safe_mask=(True,) * 5,
        completed=False,
        replanned=False,
    )

    service._publish_decision(decision, episode=0, step=1)

    assert output.throttle == (-40, 20, 90.0, 10, 0.3)
    assert action_bridge.command == (-40, 20)


def test_reset_wait_requires_current_request_transition(monkeypatch):
    output = _OutputCapture()
    output.device_data = SimpleNamespace(
        task_status=2,
        reset_status=2,
    )
    transitions = iter((1, 2))

    def advance_reset(_duration):
        output.device_data.reset_status = next(transitions)

    monkeypatch.setattr(
        "usvlib4ros.navigation.fixed_map_service.time.sleep",
        advance_reset,
    )
    service = FixedMapNavigationService(object(), output)

    completed = service._wait_for_reset(timeout_s=1.0)

    assert completed
    assert output.throttle_updates == 2


def test_reset_wait_accepts_fast_completion_after_known_baseline():
    output = _OutputCapture()
    output.device_data = SimpleNamespace(
        task_status=2,
        reset_status=2,
        reset_request_time=11.0,
    )
    service = FixedMapNavigationService(object(), output)

    completed = service._wait_for_reset(
        timeout_s=0.01,
        initial_request_time=10.0,
    )

    assert completed


def test_unpromoted_checkpoint_waits_for_new_train_trigger_without_reset_loop(
    monkeypatch,
    capsys,
):
    class _StopService(Exception):
        pass

    class _RosCapture:
        def __init__(self):
            self.reset_calls = 0

        def initParameterList(self):
            return None

        def reset_unity(self):
            self.reset_calls += 1
            return True

        def set_auto_work(self):
            return True

        def getRoute(self):
            return SimpleNamespace(
                name="National_Test",
                points=[object()] * 13,
                obstacles=[object()] * 16,
            )

    ros = _RosCapture()
    output = _OutputCapture()
    output.device_data = SimpleNamespace(task_status=1)
    service = FixedMapNavigationService(ros, output)
    monkeypatch.setattr(service, "_wait_for_reset", lambda **_: True)
    monkeypatch.setattr(service, "_wait_for_auto", lambda **_: True)
    monkeypatch.setattr(
        service,
        "_run_episode",
        lambda *_: (_ for _ in ()).throw(
            ValueError(
                "SAC checkpoint has not passed offline and Unity promotion"
            )
        ),
    )
    final_sleeps = 0

    def advance_service(duration):
        nonlocal final_sleeps
        if duration == 1.0:
            if output.device_data.task_status == 0:
                raise _StopService
            output.device_data.task_status = 0
        elif duration == 0.02:
            final_sleeps += 1
            if final_sleeps >= 2:
                raise _StopService

    monkeypatch.setattr(
        "usvlib4ros.navigation.fixed_map_service.time.sleep",
        advance_service,
    )

    try:
        service.run()
    except _StopService:
        pass

    assert ros.reset_calls == 1
    assert output.throttle[:2] == (0, 0)
    output_text = capsys.readouterr().out
    assert "Route National_Test: 13 points, 16 obstacles" in output_text
    assert "namespace(" not in output_text
    assert "ROS/Unity connection OK" in output_text


def test_single_episode_completion_holds_zero_and_stops_reset_loop(
    monkeypatch,
):
    class _StopService(Exception):
        pass

    class _RosCapture:
        def __init__(self):
            self.reset_calls = 0

        def initParameterList(self):
            return None

        def reset_unity(self):
            self.reset_calls += 1
            return True

        def set_auto_work(self):
            return True

        def getRoute(self):
            return SimpleNamespace(
                name="National_Test",
                points=[object()] * 13,
                obstacles=[object()] * 16,
            )

    ros = _RosCapture()
    output = _OutputCapture()
    output.device_data = SimpleNamespace(task_status=1)
    service = FixedMapNavigationService(ros, output)
    monkeypatch.setattr(service, "_wait_for_reset", lambda **_: True)
    monkeypatch.setattr(service, "_wait_for_auto", lambda **_: True)
    monkeypatch.setattr(service, "_run_episode", lambda *_, **__: True)
    hold_samples = 0

    def advance_service(duration):
        nonlocal hold_samples
        if duration == 1.0:
            if output.device_data.task_status == 0:
                raise _StopService
            output.device_data.task_status = 0
        elif duration == 0.1:
            hold_samples += 1
            if hold_samples >= 3:
                output.device_data.task_status = 0
        elif duration == 0.02:
            raise _StopService

    monkeypatch.setattr(
        "usvlib4ros.navigation.fixed_map_service.time.sleep",
        advance_service,
    )

    try:
        service.run()
    except _StopService:
        pass

    assert ros.reset_calls == 1
    assert hold_samples >= 3
    assert output.throttle[:2] == (0, 0)


def test_empty_unity_route_uses_approved_fixed_route_without_reset_loop(
    monkeypatch,
    tmp_path,
    capsys,
):
    class _StopService(Exception):
        pass

    class _RosCapture:
        def __init__(self):
            self.reset_calls = 0

        def initParameterList(self):
            return None

        def reset_unity(self):
            self.reset_calls += 1
            return True

        def set_auto_work(self):
            return True

        def getRoute(self):
            return SimpleNamespace(
                id="",
                name="",
                version=0,
                start_index=0,
                points=[],
                obstacles=[],
            )

    ros = _RosCapture()
    output = _OutputCapture()
    output.device_data = SimpleNamespace(task_status=1)
    service = FixedMapNavigationService(
        ros,
        output,
        policy_mode=PolicyMode.UNITY_TEST,
        reports_dir=tmp_path,
    )
    monkeypatch.setattr(service, "_wait_for_reset", lambda **_: True)
    monkeypatch.setattr(service, "_wait_for_auto", lambda **_: True)
    episodes = []

    def run_episode(route, episode, **_kwargs):
        episodes.append((route, episode))
        return True

    monkeypatch.setattr(service, "_run_episode", run_episode)

    def advance_service(duration):
        if duration == 0.1:
            output.device_data.task_status = 0
        elif duration == 1.0:
            if output.device_data.task_status == 0:
                raise _StopService
            output.device_data.task_status = 0
        elif duration == 0.02:
            raise _StopService

    monkeypatch.setattr(
        "usvlib4ros.navigation.fixed_map_service.time.sleep",
        advance_service,
    )

    try:
        service.run()
    except _StopService:
        pass

    assert ros.reset_calls == 1
    assert len(episodes) == 1
    route, episode = episodes[0]
    assert episode == 0
    assert route.id
    assert route.name == "National_Test"
    assert len(route.points) == 13
    pose = SimpleNamespace(
        lat=route.points[0].lat,
        lng=route.points[0].lng,
    )
    context = build_live_route_context(
        route,
        pose,
        session_id="approved-fallback-test",
    )
    assert context.fit_residual_m < 0.05
    assert "approved fixed route fallback" in capsys.readouterr().out
