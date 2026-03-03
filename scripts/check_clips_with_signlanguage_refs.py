import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.english_to_asl.dictionary.asl_signs import ASL_SIGNS


def parse_args():
    p = argparse.ArgumentParser(
        description="Cross-check local clips against SignLanguage reference titles/URLs."
    )
    p.add_argument(
        "--refs-json",
        type=Path,
        default=Path("data/external/SignLanguage/frontend/utils/signLanguageVideo.json"),
        help="Path to SignLanguage reference JSON with title+URL entries.",
    )
    p.add_argument(
        "--clips-dir",
        type=Path,
        default=Path("ui/animation/clips"),
    )
    p.add_argument(
        "--max-seconds",
        type=float,
        default=10.0,
        help="Flag clip variants longer than this duration.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/eval/signlanguage_reference_check.json"),
    )
    return p.parse_args()


def norm_text(value: str) -> str:
    value = value.strip().lower().replace("_", " ")
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def token_from_clip_name(stem: str) -> str:
    parts = stem.lower().split("_")
    out = []
    for p in parts:
        if p.isdigit():
            break
        out.append(p)
    return "_".join(out).upper() if out else stem.upper()


def load_ref_entries(path: Path) -> List[Dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for row in rows:
        title = str(row.get("title", "")).strip()
        url = str(row.get("URL", "")).strip()
        if not title or not url:
            continue
        out.append(
            {
                "title": title,
                "url": url,
                "title_norm": norm_text(title),
            }
        )
    return out


def find_refs_for_token(token: str, refs: List[Dict]) -> Dict[str, List[Dict]]:
    t = norm_text(token)
    exact = []
    prefix = []
    contains = []
    word_pat = re.compile(rf"(?:^|\s){re.escape(t)}(?:\s|$)")
    for row in refs:
        n = row["title_norm"]
        if n == t:
            exact.append(row)
        elif n.startswith(t + " "):
            prefix.append(row)
        elif word_pat.search(n):
            contains.append(row)
    return {
        "exact": exact,
        "prefix": prefix,
        "contains": contains,
    }


def main():
    args = parse_args()
    if not args.refs_json.exists():
        raise FileNotFoundError(f"Reference JSON not found: {args.refs_json}")
    if not args.clips_dir.exists():
        raise FileNotFoundError(f"Clips dir not found: {args.clips_dir}")

    refs = load_ref_entries(args.refs_json)

    token_to_clips: Dict[str, List[Dict]] = {}
    for fp in sorted(args.clips_dir.glob("*.json")):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            fps = float(data.get("fps", 0) or 0)
            frames = data.get("frames", [])
            if fps <= 0 or not isinstance(frames, list) or not frames:
                continue
            duration = len(frames) / fps
            token = token_from_clip_name(fp.stem)
            token_to_clips.setdefault(token, []).append(
                {
                    "clip": fp.name,
                    "duration": round(duration, 3),
                    "frames": len(frames),
                    "fps": fps,
                }
            )
        except Exception:
            continue

    # Cover both dictionary tokens and discovered clip tokens.
    tokens = sorted(set(ASL_SIGNS.keys()) | set(token_to_clips.keys()))
    checks = []
    suspicious = []
    for token in tokens:
        clips = token_to_clips.get(token, [])
        refs_for_token = find_refs_for_token(token, refs)
        exact_count = len(refs_for_token["exact"])
        prefix_count = len(refs_for_token["prefix"])
        contains_count = len(refs_for_token["contains"])
        worst_duration = max((c["duration"] for c in clips), default=0.0)
        long_clip_names = [c["clip"] for c in clips if c["duration"] > args.max_seconds]

        status = "ok"
        reasons = []
        if clips and long_clip_names:
            status = "suspicious"
            reasons.append("long_clip_duration")
        if clips and (exact_count + prefix_count + contains_count == 0):
            status = "suspicious"
            reasons.append("no_reference_match")

        item = {
            "token": token,
            "clip_count": len(clips),
            "worst_duration_seconds": round(worst_duration, 3),
            "long_clips": long_clip_names,
            "reference_counts": {
                "exact": exact_count,
                "prefix": prefix_count,
                "contains": contains_count,
            },
            "reference_examples": {
                "exact": refs_for_token["exact"][:3],
                "prefix": refs_for_token["prefix"][:3],
                "contains": refs_for_token["contains"][:3],
            },
            "status": status,
            "reasons": reasons,
        }
        checks.append(item)
        if status != "ok":
            suspicious.append(item)

    report = {
        "refs_json": str(args.refs_json),
        "clips_dir": str(args.clips_dir),
        "tokens_checked": len(checks),
        "suspicious_count": len(suspicious),
        "suspicious": suspicious,
        "checks": checks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"tokens_checked={len(checks)}")
    print(f"suspicious_count={len(suspicious)}")
    print(f"report={args.out}")
    if suspicious:
        print("Top suspicious tokens:")
        for row in suspicious[:25]:
            print(
                f"  {row['token']} | clips={row['clip_count']} | "
                f"worst={row['worst_duration_seconds']}s | reasons={','.join(row['reasons'])}"
            )


if __name__ == "__main__":
    main()
