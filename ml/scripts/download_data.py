"""Download every raw input the project uses into ml/data/raw.

Sources (all public, no login):
  - WLASL index (WLASL_v0.3.json) from the official GitHub repo (dxli94/WLASL)
  - WLASL videos from the Hugging Face mirror Voxel51/WLASL (same videos as Kaggle wlasl-processed)
  - ASL Alphabet images from Kaggle (grassknoted/asl-alphabet), only the ~900 images used
  - J. Schuyler Long, "The Sign Language: A Manual of Signs" (1910), scanned PDF from archive.org

Usage:  python scripts/download_data.py [wlasl|alphabet|pdf|all] [--top 200]
"""
import argparse
import csv
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

WLASL_INDEX = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
HF_BASE = "https://huggingface.co/datasets/Voxel51/WLASL/resolve/main/"
ALPHABET = "https://www.kaggle.com/api/v1/datasets/download/grassknoted/asl-alphabet"
LONG_PDF = "https://archive.org/download/signlanguagemanu00long/signlanguagemanu00long.pdf"


def fetch(url, dest, retries=4):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "hum-kalaam-research/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.replace(dest)
            return "ok"
        except Exception as e:  # network hiccups: wait and retry
            if attempt == retries - 1:
                return f"failed: {e}"
            time.sleep(2 * (attempt + 1))


# The 20 words the app's camera recognises (src/camera/core-words.js), as WLASL glosses.
# Nine of them are not among the 200 most common glosses, so they are added.
APP_CAMERA_GLOSSES = ["hello", "thank you", "please", "sorry", "yes", "no", "help", "water", "eat", "sad", "more",
                      "name", "friend", "doctor", "go", "want", "sick", "tired", "understand", "again"]


def download_wlasl(top, only_app=False):
    out = RAW / "wlasl"
    print("index:", fetch(WLASL_INDEX, out / "WLASL_v0.3.json"))
    print("mirror list:", fetch(HF_BASE + "samples.json", out / "samples.json"))
    index = json.loads((out / "WLASL_v0.3.json").read_text())
    mirror = json.loads((out / "samples.json").read_text())["samples"]
    path_of = {Path(s["filepath"]).stem: s["filepath"] for s in mirror}

    rows = []
    for rank, entry in enumerate(index):
        if rank >= top and entry["gloss"] not in APP_CAMERA_GLOSSES:
            continue
        for inst in entry["instances"]:
            if inst["video_id"] in path_of:
                rows.append({
                    "video_id": inst["video_id"],
                    "gloss": entry["gloss"],
                    "gloss_rank": rank,
                    "signer_id": inst["signer_id"],
                    "split": inst["split"],
                    "fps": inst["fps"],
                    "source_url": inst["url"],
                    "mirror_path": path_of[inst["video_id"]],
                })
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} videos: the top {top} glosses plus the app's camera words")
    if only_app:
        rows = [r for r in rows if r["gloss"] in APP_CAMERA_GLOSSES]

    def one(row):
        return fetch(HF_BASE + row["mirror_path"], out / "videos" / f"{row['video_id']}.mp4")

    done = 0
    failed = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for row, status in zip(rows, pool.map(one, rows)):
            done += 1
            if status.startswith("failed"):
                failed.append((row["video_id"], status))
            if done % 100 == 0:
                print(f"  {done}/{len(rows)}", flush=True)
    print("failed:", len(failed), failed[:5])


LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]


def alphabet_plan(seed=361):
    """Which images to take from the 3,000 per letter, and what each is for.

    Each letter folder is frames of one recording (A1 ... A3000), so neighbouring
    numbers are near-duplicates. The three uses come from separate stretches of
    the recording, so a test image is never a neighbour of a training image:
        labelled    frames    1-1500  (500 images, 19-20 per letter)
        unlabelled  frames 1801-2200  (100 images; their labels are never used for training)
        test        frames 2501-3000  (10 per letter, 260 images)
    plus Kaggle's own test images (one per letter, taken separately).
    """
    import random
    rng = random.Random(seed)
    plan = []
    per_letter = [20 if i < 500 - 19 * 26 else 19 for i in range(26)]  # 6 letters get 20, the rest 19 -> 500
    for letter, n in zip(LETTERS, per_letter):
        for k in rng.sample(range(1, 1501), n):
            plan.append(("labelled", letter, k))
        for k in rng.sample(range(2501, 3001), 10):
            plan.append(("test", letter, k))
    pool = [(letter, k) for letter in LETTERS for k in range(1801, 2201)]
    for letter, k in rng.sample(pool, 100):
        plan.append(("unlabelled", letter, k))
    return plan


def download_alphabet():
    """Download only the images the project uses, one by one (the whole zip is 1.1 GB)."""
    from urllib.parse import quote
    out = RAW / "asl_alphabet"
    rows = []
    for use, letter, k in alphabet_plan():
        rel = f"asl_alphabet_train/asl_alphabet_train/{letter}/{letter}{k}.jpg"
        rows.append({"use": use, "letter": letter, "kaggle_path": rel, "file": f"{use}/{letter}{k}.jpg"})
    for letter in LETTERS:
        rel = f"asl_alphabet_test/asl_alphabet_test/{letter}_test.jpg"
        rows.append({"use": "kaggle_test", "letter": letter, "kaggle_path": rel, "file": f"kaggle_test/{letter}_test.jpg"})

    def one(row):
        return fetch(f"{ALPHABET}/{quote(row['kaggle_path'], safe='')}", out / row["file"])

    with ThreadPoolExecutor(max_workers=12) as pool:
        statuses = list(pool.map(one, rows))
    failed = [r["file"] for r, s in zip(rows, statuses) if s.startswith("failed")]
    # The letter of an unlabelled image is kept in the manifest only to score the
    # pseudo-labels afterwards; training code must not read it.
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(r for r in rows if r["file"] not in failed)
    print(f"{len(rows) - len(failed)} images, {len(failed)} failed", failed[:5])


def download_pdf():
    print("pdf:", fetch(LONG_PDF, RAW / "long_1910" / "sign_language_manual_long_1910.pdf"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["wlasl", "alphabet", "pdf", "all"])
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--only-app", action="store_true", help="only the videos of the app's 20 camera words")
    a = ap.parse_args()
    if a.what in ("wlasl", "all"):
        download_wlasl(a.top, a.only_app)
    if a.what in ("alphabet", "all"):
        download_alphabet()
    if a.what in ("pdf", "all"):
        download_pdf()
    sys.exit(0)
