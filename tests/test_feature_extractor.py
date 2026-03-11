from core.asl_to_english.features import pose_to_feature_vector


def _pose(x_shift: float = 0.0):
    pose = {
        "head": (0.5 + x_shift, 0.2),
        "shoulder_left": (0.4 + x_shift, 0.4),
        "elbow_left": (0.35 + x_shift, 0.55),
        "hand_left": (0.3 + x_shift, 0.7),
        "shoulder_right": (0.6 + x_shift, 0.4),
        "elbow_right": (0.65 + x_shift, 0.55),
        "hand_right": (0.7 + x_shift, 0.7),
        "torso": (0.5 + x_shift, 0.7),
    }
    for idx in range(21):
        pose[f"left_hand_{idx}"] = (0.28 + x_shift + (idx * 0.001), 0.68 + (idx * 0.001))
        pose[f"right_hand_{idx}"] = (0.68 + x_shift + (idx * 0.001), 0.68 + (idx * 0.001))
    pose["left_thumb_tip"] = pose["left_hand_4"]
    pose["left_index_tip"] = pose["left_hand_8"]
    pose["left_middle_tip"] = pose["left_hand_12"]
    pose["left_ring_tip"] = pose["left_hand_16"]
    pose["left_pinky_tip"] = pose["left_hand_20"]
    pose["right_thumb_tip"] = pose["right_hand_4"]
    pose["right_index_tip"] = pose["right_hand_8"]
    pose["right_middle_tip"] = pose["right_hand_12"]
    pose["right_ring_tip"] = pose["right_hand_16"]
    pose["right_pinky_tip"] = pose["right_hand_20"]
    return pose


def test_pose_to_feature_vector_has_expected_size_and_motion_component():
    first = pose_to_feature_vector(_pose(0.0))
    moved = _pose(0.0)
    moved["right_index_tip"] = (moved["right_index_tip"][0] + 0.05, moved["right_index_tip"][1])
    moved["right_hand_8"] = moved["right_index_tip"]
    second = pose_to_feature_vector(moved, prev_pose=_pose(0.0))

    assert first.shape == second.shape
    assert first.shape[0] > 100
    assert float(abs(second - first).sum()) > 0.0
