import argparse
import json
from pathlib import Path
import sys
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.engine import english_to_asl


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate English -> ASL token output on a labeled sentence set."
    )
    parser.add_argument(
        "--eval-file",
        default="data/eval/english_to_asl_eval.json",
        help="Path to JSON array with fields: text, expected_tokens.",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=10,
        help="Max number of failing examples to print.",
    )
    return parser.parse_args()


def _token_counts(pred: List[str], expected: List[str]) -> tuple[int, int, int]:
    matched = 0
    remaining = list(expected)
    for token in pred:
        if token in remaining:
            matched += 1
            remaining.remove(token)
    return matched, len(pred), len(expected)


def main():
    args = parse_args()
    eval_path = Path(args.eval_file)
    rows = json.loads(eval_path.read_text(encoding="utf-8"))

    total = len(rows)
    exact = 0
    total_matched = 0
    total_pred = 0
    total_expected = 0
    failures = []

    for row in rows:
        text = row["text"]
        expected = [str(t).upper() for t in row["expected_tokens"]]
        result = english_to_asl(text=text)
        pred = result.asl_tokens

        if pred == expected:
            exact += 1
        else:
            failures.append(
                {
                    "text": text,
                    "expected": expected,
                    "predicted": pred,
                }
            )

        matched, pred_count, expected_count = _token_counts(pred, expected)
        total_matched += matched
        total_pred += pred_count
        total_expected += expected_count

    exact_acc = exact / total if total else 0.0
    precision = total_matched / total_pred if total_pred else 0.0
    recall = total_matched / total_expected if total_expected else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    print(f"[eval] file={eval_path}")
    print(f"[eval] samples={total}")
    print(f"[eval] exact_match={exact_acc:.3f} ({exact}/{total})")
    print(f"[eval] token_precision={precision:.3f}")
    print(f"[eval] token_recall={recall:.3f}")
    print(f"[eval] token_f1={f1:.3f}")

    if failures:
        print(f"[eval] failures={len(failures)}")
        for i, failure in enumerate(failures[: max(0, args.max_failures)], start=1):
            print(f"[fail {i}] text={failure['text']}")
            print(f"  expected={failure['expected']}")
            print(f"  predicted={failure['predicted']}")
    else:
        print("[eval] failures=0")


if __name__ == "__main__":
    main()
