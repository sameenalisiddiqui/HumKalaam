"""Run MediaPipe on every WLASL video and save the raw landmarks, one row per frame.

The two models are the app's own (public/models/hand_landmarker.task and
pose_landmarker_lite.task) with the app's settings (camera-tracker.js), so the
numbers a model is trained on are the numbers the camera produces in the browser.

Nothing is cleaned here. A hand MediaPipe did not find is left empty (NaN), and
hands are stored in the order MediaPipe returned them, with its own Left/Right
label and score. Cleaning (assigning hands to the body, filling gaps, smoothing)
happens later, in clean_landmarks.py.

Output
  data/processed/landmarks/<video_id>.npz   one file per video (resumable)
  data/processed/wlasl_landmarks_raw.csv.gz  every frame of every video, one row each
"""
import argparse
import csv
import gzip
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT.parent
RAW = ROOT / "data" / "raw" / "wlasl"
OUT = ROOT / "data" / "processed" / "landmarks"
HAND_MODEL = APP / "public" / "models" / "hand_landmarker.task"
POSE_MODEL = APP / "public" / "models" / "pose_landmarker_lite.task"

N_POSE, N_HAND = 33, 21
# The body model runs on every second frame (shoulders and elbows move slowly;
# the hands, which move fast, run on every frame). On this laptop's CPU that
# saves about a fifth of the time. The skipped frames are left empty here and
# filled by interpolation in clean_landmarks.py.
POSE_EVERY = 2
# Per frame: pose (33 x x,y,z,visibility), then two hand slots, each
# [label(0=Left,1=Right), score, 21 x image x,y,z, 21 x world x,y,z].
HAND_W = 2 + N_HAND * 3 * 2
WIDTH = N_POSE * 4 + 2 * HAND_W


def make_landmarkers():
    import mediapipe as mp
    v = mp.tasks.vision
    base = mp.tasks.BaseOptions
    hands = v.HandLandmarker.create_from_options(v.HandLandmarkerOptions(
        base_options=base(model_asset_path=str(HAND_MODEL)), running_mode=v.RunningMode.VIDEO,
        num_hands=2, min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5))
    pose = v.PoseLandmarker.create_from_options(v.PoseLandmarkerOptions(
        base_options=base(model_asset_path=str(POSE_MODEL)), running_mode=v.RunningMode.VIDEO,
        num_poses=1, min_pose_detection_confidence=0.5, min_tracking_confidence=0.5))
    return hands, pose


def extract_video(video_id):
    """Landmarks for every frame of one video -> (frames x WIDTH) float32, plus fps and size."""
    dest = OUT / f"{video_id}.npz"
    if dest.exists():
        return video_id, "cached"
    import cv2
    import mediapipe as mp
    cap = cv2.VideoCapture(str(RAW / "videos" / f"{video_id}.mp4"))
    if not cap.isOpened():
        return video_id, "unreadable"
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    hands, pose = make_landmarkers()  # fresh trackers per video: tracking must not carry over
    rows = []
    i = 0
    w = h = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(round(i * 1000.0 / fps)) + i  # strictly increasing, as VIDEO mode requires
        hr = hands.detect_for_video(img, ts)
        pr = pose.detect_for_video(img, ts) if i % POSE_EVERY == 0 else None
        row = np.full(WIDTH, np.nan, dtype=np.float32)
        if pr is not None and pr.pose_landmarks:
            for j, q in enumerate(pr.pose_landmarks[0]):
                row[j * 4:j * 4 + 4] = (q.x, q.y, q.z, q.visibility if q.visibility is not None else np.nan)
        for slot, lm in enumerate(hr.hand_landmarks[:2]):
            base_i = N_POSE * 4 + slot * HAND_W
            cat = hr.handedness[slot][0]
            row[base_i] = 0.0 if cat.category_name == "Left" else 1.0
            row[base_i + 1] = cat.score
            row[base_i + 2:base_i + 2 + 63] = np.array([(q.x, q.y, q.z) for q in lm], dtype=np.float32).ravel()
            world = hr.hand_world_landmarks[slot]
            row[base_i + 2 + 63:base_i + HAND_W] = np.array([(q.x, q.y, q.z) for q in world], dtype=np.float32).ravel()
        rows.append(row)
        i += 1
    cap.release()
    hands.close()
    pose.close()
    if not rows:
        return video_id, "no frames"
    tmp = dest.with_name(dest.stem + ".part.npz")  # written whole, then renamed, so a stopped run leaves no half file
    np.savez_compressed(tmp, lm=np.stack(rows), fps=fps, size=np.array([w, h]), pose_every=POSE_EVERY)
    tmp.replace(dest)
    return video_id, f"{len(rows)} frames"


def column_names():
    cols = []
    for j in range(N_POSE):
        cols += [f"pose{j}_{a}" for a in ("x", "y", "z", "vis")]
    for s in range(2):
        cols += [f"hand{s}_label", f"hand{s}_score"]
        for kind in ("img", "world"):
            for k in range(N_HAND):
                cols += [f"hand{s}_{kind}{k}_{a}" for a in ("x", "y", "z")]
    assert len(cols) == WIDTH
    return cols


def write_csv(manifest):
    path = ROOT / "data" / "processed" / "wlasl_landmarks_raw.csv.gz"
    header = ["video_id", "gloss", "signer_id", "split", "frame", "time_ms", "width", "height"] + column_names()
    n = 0
    with gzip.open(path, "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for m in manifest:
            p = OUT / f"{m['video_id']}.npz"
            if not p.exists():
                continue
            z = np.load(p)
            fps = float(z["fps"])
            width, height = (int(v) for v in z["size"])
            for i, row in enumerate(z["lm"]):
                vals = ["" if np.isnan(v) else f"{v:.5f}" for v in row]
                w.writerow([m["video_id"], m["gloss"], m["signer_id"], m["split"], i, round(i * 1000 / fps), width, height] + vals)
                n += 1
    print(f"wrote {n} rows x {len(header)} columns to {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # Two workers measured fastest on this 4-core laptop (4.8 frames/s, against 3.5 with one and 3.2 with three).
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--csv-only", action="store_true")
    ap.add_argument("--no-csv", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(RAW / "manifest.csv", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    if not a.csv_only:
        from download_data import APP_CAMERA_GLOSSES
        # The app's camera words first, then the most common glosses, so models can start on partial data.
        todo = sorted(manifest, key=lambda m: (m["gloss"] not in APP_CAMERA_GLOSSES, int(m["gloss_rank"])))
        ids = [m["video_id"] for m in todo if (RAW / "videos" / f"{m['video_id']}.mp4").exists()]
        print(f"{len(ids)} videos, {a.workers} workers", flush=True)
        done = 0
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for vid, status in pool.map(extract_video, ids, chunksize=4):
                done += 1
                if not status.endswith("frames") and status != "cached":
                    print(vid, status, flush=True)
                if done % 100 == 0:
                    print(f"  {done}/{len(ids)}", flush=True)
    if not a.no_csv:
        write_csv(manifest)
