"""Raw landmarks of the app's 20 camera words, for the JavaScript side.

  data/processed/app20_raw.json   every app20 video: raw rows (as the camera would give
                                   them) and the Python-cleaned features. Used by
                                   eval_template_recognizer.mjs to score the app's
                                   original (avatar-template) recogniser on real signers.
  ../tests/fixtures/ml-app20.json  three test videos with their Python features and
                                   model probabilities, for tests/ml-features.test.mjs
                                   (the JS port must give the same numbers).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
sys.path.insert(0, str(Path(__file__).parent))
from clean_landmarks import clean_video  # noqa: E402
from tabular_models import APP20  # noqa: E402


def to_list(a):
    # Full precision: rounding the raw landmarks can tip a tree split in the XGBoost model.
    return [[None if np.isnan(v) else float(v) for v in row] for row in a]


def main():
    videos = pd.read_csv(PROC / "videos.csv", dtype={"video_id": str})
    names = json.loads((PROC / "feature_names.json").read_text())
    app = videos[videos["gloss"].isin(APP20)]
    out = []
    for _, v in app.iterrows():
        z = np.load(PROC / "landmarks" / f"{v['video_id']}.npz")
        size = [int(s) for s in z["size"]]
        pose_every = int(z["pose_every"]) if "pose_every" in z else 1
        lm = z["lm"].astype(np.float64)
        res = clean_video(lm, tuple(size), __import__("collections").Counter(), pose_every)
        if res is None:
            continue
        frame, dom = res
        out.append({"video_id": v["video_id"], "gloss": v["gloss"], "split": v["split"], "signer_id": int(v["signer_id"]),
                    "size": size, "dominant": dom, "raw": to_list(lm), "features": to_list(frame[names].to_numpy())})
    (PROC / "app20_raw.json").write_text(json.dumps({"feature_names": names, "videos": out}))
    print(f"{len(out)} app20 videos -> {PROC / 'app20_raw.json'}")

    # Small fixture for the unit test: three test videos, with the trained models' probabilities when they exist.
    preds_path = ROOT / "results" / "tabular_app20_predictions.json"
    preds = json.loads(preds_path.read_text()) if preds_path.exists() else None
    test = [v for v in out if v["split"] == "test"][:3]
    for v in test:
        if preds and v["video_id"] in preds["video_ids"]:
            i = preds["video_ids"].index(v["video_id"])
            v["xgb_video"] = preds["xgb_video"][i]
            v["gru"] = preds["gru"][i]
    fx = ROOT.parent / "tests" / "fixtures"
    fx.mkdir(parents=True, exist_ok=True)
    (fx / "ml-app20.json").write_text(json.dumps({"feature_names": names, "classes": preds["classes"] if preds else None, "videos": test}))
    print(f"{len(test)} fixture videos -> {fx / 'ml-app20.json'}")


if __name__ == "__main__":
    main()
