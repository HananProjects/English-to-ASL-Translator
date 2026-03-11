from core.vision.live_pose import LivePoseFilter


def test_live_pose_filter_smooths_points():
    pose_filter = LivePoseFilter(alpha=0.5, hand_hold_frames=1, body_hold_frames=1)

    first = pose_filter.apply({"hand_right": (0.2, 0.4)})
    second = pose_filter.apply({"hand_right": (0.6, 0.8)})

    assert first["hand_right"] == (0.2, 0.4)
    assert second["hand_right"] == (0.4, 0.6000000000000001)


def test_live_pose_filter_holds_hand_points_briefly_when_tracking_drops():
    pose_filter = LivePoseFilter(alpha=1.0, hand_hold_frames=2, body_hold_frames=0)

    pose_filter.apply({"hand_right": (0.3, 0.5), "right_index_tip": (0.35, 0.45)})
    held = pose_filter.apply({})
    still_held = pose_filter.apply({})
    expired = pose_filter.apply({})

    assert held == {}
    assert still_held == {}
    assert expired == {}

    pose_filter.reset()
    pose_filter.apply({"hand_right": (0.3, 0.5), "right_index_tip": (0.35, 0.45)})
    held_with_body = pose_filter.apply({"shoulder_right": (0.6, 0.4)})
    assert held_with_body["hand_right"] == (0.3, 0.5)
    assert held_with_body["right_index_tip"] == (0.35, 0.45)
