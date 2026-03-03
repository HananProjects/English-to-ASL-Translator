import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(
        description="Train general-vocabulary ASL model using existing sequence trainer."
    )
    p.add_argument(
        "--labels-file",
        type=Path,
        default=REPO_ROOT / "data" / "labels_general.txt",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "models" / "asl_landmark_classifier_general.npz",
    )
    p.add_argument("--clips-dir", type=Path, default=REPO_ROOT / "ui" / "animation" / "clips")
    p.add_argument("--seq-len", type=int, default=24)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--l2", type=float, default=0.005)
    p.add_argument("--min-samples-per-label", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    trainer = REPO_ROOT / "scripts" / "train_asl_sequence_model.py"
    cmd = [
        sys.executable,
        str(trainer),
        "--clips-dir",
        str(args.clips_dir),
        "--labels-file",
        str(args.labels_file),
        "--min-samples-per-label",
        str(args.min_samples_per_label),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--l2",
        str(args.l2),
        "--seq-len",
        str(args.seq_len),
        "--out",
        str(args.out),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()

