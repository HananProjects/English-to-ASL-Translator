import json
from pathlib import Path

from core.asl_to_english.dataset_utils import infer_label_from_clip_name


def test_infer_label_from_clip_name_handles_numeric_suffix():
    assert infer_label_from_clip_name("hello_1") == "HELLO"
    assert infer_label_from_clip_name("thank_you_1") == "THANK_YOU"


def test_manifest_shape_example(tmp_path: Path):
    out = tmp_path / "manifest.json"
    report = {
        "clips_dir": "ui/animation/clips",
        "labels_file": "data/labels_100.txt",
        "total_clips": 2,
        "total_labels": 2,
        "min_samples_per_label": 2,
        "sparse_labels": ["HELP"],
        "coverage": [{"label": "HELP", "count": 1, "unique_signers": 0}],
        "samples": [{"clip": "help_1", "path": "ui/animation/clips/help_1.json", "label": "HELP"}],
    }
    out.write_text(json.dumps(report), encoding="utf-8")

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["total_clips"] == 2
    assert loaded["coverage"][0]["label"] == "HELP"
