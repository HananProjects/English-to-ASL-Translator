import numpy as np

from core.asl_to_english.features import pose_to_feature_vector
from core.asl_to_english.recognizer import (
    JOINT_KEYS,
    ModelMatcher,
    SignStreamRecognizer,
    normalize_pose,
    pose_distance,
)


class DummyMatcher:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.index = 0

    def match(self, pose):
        if self.index >= len(self.outputs):
            return (None, 0.0)
        value = self.outputs[self.index]
        self.index += 1
        return value


def test_normalize_pose_is_position_and_scale_robust():
    pose_a = {
        "torso": (0.5, 0.5),
        "shoulder_left": (0.4, 0.5),
        "shoulder_right": (0.6, 0.5),
        "hand_right": (0.7, 0.4),
        "hand_left": (0.3, 0.4),
    }
    pose_b = {
        "torso": (0.2, 0.2),
        "shoulder_left": (0.0, 0.2),
        "shoulder_right": (0.4, 0.2),
        "hand_right": (0.6, 0.0),
        "hand_left": (-0.2, 0.0),
    }
    norm_a = normalize_pose(pose_a)
    norm_b = normalize_pose(pose_b)
    assert pose_distance(norm_a, norm_b, min_shared=4) < 1e-6


def test_sign_stream_recognizer_emits_stable_tokens_and_segments():
    matcher = DummyMatcher(
        [
            ("YOU", 0.9), ("YOU", 0.9), ("YOU", 0.9),
            ("GO", 0.9), ("GO", 0.9), ("GO", 0.9),
            ("WHERE", 0.9), ("WHERE", 0.9), ("WHERE", 0.9),
            (None, 0.0), (None, 0.0), (None, 0.0), (None, 0.0),
        ]
    )
    recognizer = SignStreamRecognizer(
        matcher=matcher,
        stable_frames=3,
        min_confidence=0.5,
        emit_cooldown_frames=0,
        pause_frames=4,
    )

    emitted = []
    sentence = None
    for _ in range(13):
        update = recognizer.process({})
        if update.detected_token is not None:
            emitted.append(update.detected_token)
        if update.sentence_tokens is not None:
            sentence = update.sentence_tokens

    assert emitted == ["YOU", "GO", "WHERE"]
    assert sentence == ["YOU", "GO", "WHERE"]


def test_sign_stream_recognizer_does_not_repeat_same_token_while_held():
    matcher = DummyMatcher(
        [("YOU", 0.9)] * 10
    )
    recognizer = SignStreamRecognizer(
        matcher=matcher,
        stable_frames=3,
        min_confidence=0.5,
        emit_cooldown_frames=2,
        pause_frames=4,
    )

    emitted = []
    for _ in range(10):
        update = recognizer.process({})
        if update.detected_token is not None:
            emitted.append(update.detected_token)

    assert emitted == ["YOU"]


def test_model_matcher_predicts_from_npz_model(tmp_path):
    seq_len = 3
    feature_dim = int(pose_to_feature_vector({}).shape[0])
    classes = 2

    W = np.zeros((seq_len * feature_dim, classes), dtype=np.float32)
    b = np.array([0.0, 2.0], dtype=np.float32)
    mean = np.zeros((1, seq_len * feature_dim), dtype=np.float32)
    std = np.ones((1, seq_len * feature_dim), dtype=np.float32)
    labels = np.array(["HELLO", "GO"])

    model_path = tmp_path / "toy_model.npz"
    np.savez_compressed(
        model_path,
        W=W,
        b=b,
        mean=mean,
        std=std,
        labels=labels,
        seq_len=np.int64(seq_len),
        feature_dim=np.int64(feature_dim),
    )

    matcher = ModelMatcher(model_path)

    token, conf = matcher.match({})
    assert token is None
    assert conf == 0.0

    token, conf = matcher.match({})
    assert token is None
    assert conf == 0.0

    token, conf = matcher.match({})
    assert token == "GO"
    assert conf > 0.5
