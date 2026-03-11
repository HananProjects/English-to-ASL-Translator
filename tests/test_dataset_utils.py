from pathlib import Path

from core.asl_to_english.dataset_utils import ClipSample, infer_label_from_clip_name, split_samples


def test_infer_label_from_standard_clip_name():
    assert infer_label_from_clip_name("thank_you_03") == "THANK_YOU"


def test_split_samples_keeps_same_signer_together():
    samples = [
        ClipSample(Path("a.json"), "a", "HELLO", "alice", "s1", "team"),
        ClipSample(Path("b.json"), "b", "HELLO", "alice", "s2", "team"),
        ClipSample(Path("c.json"), "c", "HELLO", "bob", "s3", "team"),
        ClipSample(Path("d.json"), "d", "HELLO", "bob", "s4", "team"),
    ]

    train_idx, val_idx = split_samples(samples, val_ratio=0.5, seed=42, group_by_signer=True)

    assert set(train_idx + val_idx) == {0, 1, 2, 3}
    assert ({0, 1}.issubset(set(train_idx)) and {2, 3}.issubset(set(val_idx))) or (
        {2, 3}.issubset(set(train_idx)) and {0, 1}.issubset(set(val_idx))
    )


def test_split_samples_honors_explicit_manifest_splits():
    samples = [
        ClipSample(Path("a.json"), "a", "HELLO", "alice", "s1", "team", split="train"),
        ClipSample(Path("b.json"), "b", "HELLO", "bob", "s2", "team", split="val"),
    ]

    train_idx, val_idx = split_samples(samples, val_ratio=0.5, seed=42, group_by_signer=True)

    assert train_idx == [0]
    assert val_idx == [1]


def test_load_clip_samples_filters_by_split_from_manifest(tmp_path: Path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    clip_path = clips_dir / "hello_1.json"
    clip_path.write_text('{"fps": 12, "frames": [{"head": [0.5, 0.5]}]}', encoding="utf-8")

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        """
        {
          "samples": [
            {
              "clip": "hello_1",
              "path": "hello_1.json",
              "label": "HELLO",
              "split": "val"
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    from core.asl_to_english.dataset_utils import load_clip_samples

    samples = load_clip_samples(clips_dir=clips_dir, manifest_path=manifest_path, split_filter="val")

    assert len(samples) == 1
    assert samples[0].split == "val"
