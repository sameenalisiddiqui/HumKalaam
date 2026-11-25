"""Raw MediaPipe landmarks -> clean, body-relative features, one row per frame.

Cleaning steps, in order (each is counted in data/processed/cleaning_report.json):
  1. Frames with no body (pose) found: filled from neighbours when the gap is at
     most 5 frames, otherwise dropped.
  2. Which hand is which: each detected hand is given to the nearer pose wrist,
     as the app does (MediaPipe's own Left/Right label flips when the palm is hidden).
     Two detections of the same hand: the more confident one is kept.
  3. Short tracking drop-outs (at most 6 frames, 0.24 s) are filled by linear
     interpolation; longer gaps stay missing (the hand is down or out of view).
  4. Jitter: a 3-frame centred moving average over every visible stretch.
  5. Body frame: positions in shoulder widths from the middle of the shoulders,
     x toward the signer's left, y up (the app's convention). The shoulder width is
     the video's median, so one bad frame cannot rescale it.
  6. Dominant hand: the hand that travels further. Left-handed signing is mirrored
     so every video reads as right-handed; the app mirrors the same way when the
     user chooses the left hand.
  7. Handshape: the 21 hand points relative to the wrist and scaled by the palm
     length, plus the 15 distances between the wrist and fingertips.

Output
  data/processed/wlasl_features.csv.gz   one row per kept frame (the tabular dataset)
  data/processed/sequences.npz           the same features per video, for the sequence model
  data/processed/cleaning_report.json
"""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "wlasl"
LM = ROOT / "data" / "processed" / "landmarks"
OUT = ROOT / "data" / "processed"

N_POSE, HAND_W = 33, 2 + 21 * 3 * 2
P_NOSE, P_LSH, P_RSH, P_LEL, P_REL, P_LWR, P_RWR = 0, 11, 12, 13, 14, 15, 16
TIPS = [0, 4, 8, 12, 16, 20]
MAX_POSE_GAP, MAX_HAND_GAP = 5, 6


def interp_gaps(x, present, max_gap):
    """Linear interpolation of rows of x (frames x d) across gaps of at most max_gap frames."""
    x = x.copy()
    present = present.copy()
    idx = np.where(present)[0]
    filled = 0
    for a, b in zip(idx[:-1], idx[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            for k in range(1, gap + 1):
                t = k / (gap + 1)
                x[a + k] = (1 - t) * x[a] + t * x[b]
            present[a + 1:b] = True
            filled += gap
    return x, present, filled


def smooth(x, present):
    """3-frame centred moving average inside each visible stretch."""
    y = x.copy()
    n = len(x)
    for i in range(n):
        if not present[i]:
            continue
        nb = [j for j in (i - 1, i, i + 1) if 0 <= j < n and present[j]]
        y[i] = x[nb].mean(0)
    return y


def hand_shape(world):
    """(frames x 21 x 3) world points -> (frames x 78): normalised points + tip distances."""
    w = world - world[:, :1]
    scale = np.linalg.norm(w[:, 9], axis=1)[:, None, None]
    w = w / np.where(scale > 1e-6, scale, 1)
    d = np.stack([np.linalg.norm(w[:, a] - w[:, b], axis=1) for i, a in enumerate(TIPS) for b in TIPS[i + 1:]], 1)
    return np.concatenate([w.reshape(len(w), -1), d], 1)


def clean_video(lm, size, report, pose_every=1):
    width, height = size
    n = len(lm)
    report["frames_raw"] += n
    pose = lm[:, :N_POSE * 4].reshape(n, N_POSE, 4)
    pose_ok = ~np.isnan(pose[:, P_LSH, 0]) & ~np.isnan(pose[:, P_RSH, 0])
    tried = np.arange(n) % pose_every == 0  # frames the body model was run on
    report["frames_body_model_run"] += int(tried.sum())
    report["frames_without_body"] += int((tried & ~pose_ok).sum())
    flat, pose_ok2, filled = interp_gaps(np.nan_to_num(pose.reshape(n, -1)), pose_ok, MAX_POSE_GAP)
    report["body_frames_interpolated"] += filled
    pose = flat.reshape(n, N_POSE, 4)
    keep = pose_ok2
    report["frames_dropped_no_body"] += int((~keep).sum())
    if keep.sum() < 8:
        report["videos_dropped_too_few_frames"] += 1
        return None

    px = pose[..., :2] * np.array([width, height])  # pixels
    span = np.median(np.linalg.norm(px[keep, P_LSH] - px[keep, P_RSH], axis=1))
    if span < 10:
        report["videos_dropped_no_shoulders"] += 1
        return None
    centre = (px[:, P_LSH] + px[:, P_RSH]) / 2
    centre = smooth(centre, keep)

    def body(p):  # pixels -> body units (x toward signer's left = image right, y up)
        return np.stack([(p[..., 0] - centre[:, 0, None] if p.ndim == 3 else p[..., 0] - centre[:, 0]) / span,
                         -(p[..., 1] - centre[:, 1, None] if p.ndim == 3 else p[..., 1] - centre[:, 1]) / span], -1)

    # Hands: assign each detection to the nearer pose wrist.
    sides = {"L": {"img": np.full((n, 21, 3), np.nan), "world": np.full((n, 21, 3), np.nan), "score": np.zeros(n)},
             "R": {"img": np.full((n, 21, 3), np.nan), "world": np.full((n, 21, 3), np.nan), "score": np.zeros(n)}}
    wr = {"L": px[:, P_LWR], "R": px[:, P_RWR]}
    def cost(d, side, i):
        return np.linalg.norm(d[0][0, :2] * [width, height] - wr[side][i])

    for i in range(n):
        if not keep[i]:
            continue  # this frame is dropped anyway
        dets = []
        for s in range(2):
            b = N_POSE * 4 + s * HAND_W
            if np.isnan(lm[i, b + 2]):
                continue
            img = lm[i, b + 2:b + 65].reshape(21, 3)
            world = lm[i, b + 65:b + HAND_W].reshape(21, 3)
            dets.append((img, world, lm[i, b + 1], "L" if lm[i, b] == 0 else "R"))
        if not dets:
            continue
        report["hand_detections"] += len(dets)
        if len(dets) == 2 and np.linalg.norm((dets[0][0][0, :2] - dets[1][0][0, :2]) * [width, height]) < 0.25 * span:
            dets = [max(dets, key=lambda d: d[2])]  # the same hand found twice
            report["duplicate_hand_detections_removed"] += 1
        if len(dets) == 2:
            a, b2 = dets
            straight = cost(a, "L", i) + cost(b2, "R", i)
            crossed = cost(a, "R", i) + cost(b2, "L", i)
            pairs = [("L", a), ("R", b2)] if straight <= crossed else [("R", a), ("L", b2)]
        else:
            d = dets[0]
            pairs = [("L" if cost(d, "L", i) <= cost(d, "R", i) else "R", d)]
        for side, d in pairs:
            if d[3] != side:
                report["mediapipe_hand_label_overruled"] += 1
            sides[side]["img"][i], sides[side]["world"][i], sides[side]["score"][i] = d[0], d[1], d[2]

    hand = {}
    for side in ("L", "R"):
        present = ~np.isnan(sides[side]["img"][:, 0, 0])
        arr = np.concatenate([sides[side]["img"].reshape(n, -1), sides[side]["world"].reshape(n, -1)], 1)
        arr, present2, filled = interp_gaps(np.nan_to_num(arr), present, MAX_HAND_GAP)
        report["hand_frames_interpolated"] += filled
        arr = smooth(arr, present2)
        arr[~present2] = np.nan
        hand[side] = {"img": arr[:, :63].reshape(n, 21, 3), "world": arr[:, 63:].reshape(n, 21, 3), "present": present2}

    # Positions in body units. A hand's position is its wrist landmark, or the pose wrist when the hand was not found.
    pose_body = body(px)  # (n, 33, 2)
    pos = {}
    for side, pw in (("L", P_LWR), ("R", P_RWR)):
        hb = body(hand[side]["img"][:, :, :2] * [width, height])
        p = np.where(hand[side]["present"][:, None], hb[:, 0], pose_body[:, pw])
        tip = np.where(hand[side]["present"][:, None], hb[:, 8], np.nan)
        pos[side] = (p, tip)

    # Dominant hand = the one that travels further.
    def path(p):
        q = p[keep]
        return float(np.nansum(np.linalg.norm(np.diff(q, axis=0), axis=1)))
    dom, oth = ("R", "L") if path(pos["R"][0]) >= path(pos["L"][0]) else ("L", "R")
    mirror = -1.0 if dom == "L" else 1.0
    report["videos_left_dominant_mirrored" if dom == "L" else "videos_right_dominant"] += 1

    def mx(p):  # mirror x for left-dominant videos (x is measured toward the signer's left)
        q = p.copy()
        q[..., 0] *= mirror
        return q

    elbow = {"L": pose_body[:, P_LEL], "R": pose_body[:, P_REL]}
    nose = mx(pose_body[:, P_NOSE])
    feats = {}
    for role, side in (("dom", dom), ("oth", oth)):
        p, tip = pos[side]
        p, tip = mx(p), mx(tip)
        world = hand[side]["world"].copy()
        if dom == "L":
            world[..., 0] *= -1  # a mirrored left hand has the shape of a right hand
        feats[f"{role}_x"], feats[f"{role}_y"] = p[:, 0], p[:, 1]
        feats[f"{role}_tip_x"], feats[f"{role}_tip_y"] = tip[:, 0], tip[:, 1]
        e = mx(elbow[side])
        feats[f"{role}_elbow_x"], feats[f"{role}_elbow_y"] = e[:, 0], e[:, 1]
        feats[f"{role}_present"] = hand[side]["present"].astype(np.float32)
        feats[f"{role}_vx"], feats[f"{role}_vy"] = np.zeros(n), np.zeros(n)  # filled after dropping frames, below
        shape = hand_shape(np.nan_to_num(world))
        shape[~hand[side]["present"]] = np.nan
        for k in range(shape.shape[1]):
            feats[f"{role}_shape{k}"] = shape[:, k]
    feats["nose_x"], feats["nose_y"] = nose[:, 0], nose[:, 1]
    frame = pd.DataFrame(feats)
    frame.insert(0, "frame", np.arange(n))
    frame = frame[keep].reset_index(drop=True)
    # Hand speed in shoulder widths per second (WLASL is 25 fps), over the kept frames only.
    for role in ("dom", "oth"):
        for a in ("x", "y"):
            frame[f"{role}_v{a}"] = np.gradient(frame[f"{role}_{a}"].to_numpy()) * 25.0 if len(frame) > 1 else 0.0
    report["frames_kept"] += len(frame)
    report["frames_with_dominant_hand"] += int(frame["dom_present"].sum())
    return frame, dom


def main():
    report = Counter()
    with open(RAW / "manifest.csv", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    frames, seqs, meta = [], {}, []
    for m in manifest:
        p = LM / f"{m['video_id']}.npz"
        if not p.exists():
            report["videos_not_extracted"] += 1
            continue
        z = np.load(p)
        pose_every = int(z["pose_every"]) if "pose_every" in z else 1
        res = clean_video(z["lm"], tuple(int(v) for v in z["size"]), report, pose_every)
        report["videos_seen"] += 1
        if res is None:
            continue
        df, dom = res
        seqs[m["video_id"]] = df.drop(columns=["frame"]).to_numpy(np.float32)
        meta.append({**{k: m[k] for k in ("video_id", "gloss", "gloss_rank", "signer_id", "split")}, "dominant": dom, "frames": len(df)})
        df.insert(0, "split", m["split"])
        df.insert(0, "signer_id", m["signer_id"])
        df.insert(0, "gloss", m["gloss"])
        df.insert(0, "video_id", m["video_id"])
        frames.append(df)
    table = pd.concat(frames, ignore_index=True)
    table.to_csv(OUT / "wlasl_features.csv.gz", index=False, float_format="%.4f")
    feature_names = [c for c in frames[0].columns if c not in ("video_id", "gloss", "signer_id", "split", "frame")]
    np.savez_compressed(OUT / "sequences.npz", **seqs)
    pd.DataFrame(meta).to_csv(OUT / "videos.csv", index=False)
    (OUT / "feature_names.json").write_text(json.dumps(feature_names))
    report = {k: int(v) for k, v in report.items()}
    report.update({"rows": int(len(table)), "feature_columns": len(feature_names), "videos_kept": len(meta),
                   "glosses": int(table["gloss"].nunique()), "missing_cells_pct": round(float(table[feature_names].isna().mean().mean() * 100), 2)})
    (OUT / "cleaning_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
