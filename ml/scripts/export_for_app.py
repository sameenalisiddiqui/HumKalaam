"""Copy what the website needs into ../public/ml/:

  the trained models the app runs        signs_app20_*.onnx/json
  the results dashboard's data           dashboard.json (every result and cleaning report, in one file)
  the dashboard's figures                figures/*.png

Run it after any training script.
"""
import csv
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
DEST = ROOT.parent / "public" / "ml"

FILES = {
    ROOT / "models" / "signs_app20_gru.onnx": "signs_app20_gru.onnx",
    ROOT / "models" / "signs_app20_xgb.onnx": "signs_app20_xgb.onnx",
    ROOT / "models" / "signs_app20_meta.json": "signs_app20_meta.json",
}


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def drop(d, *keys):
    if d is None:
        return None
    return {k: v for k, v in d.items() if k not in keys}


def wlasl_facts():
    man = ROOT / "data" / "raw" / "wlasl" / "manifest.csv"
    if not man.exists():
        return None
    rows = list(csv.DictReader(open(man, encoding="utf-8")))
    have = [r for r in rows if (ROOT / "data" / "raw" / "wlasl" / "videos" / f"{r['video_id']}.mp4").exists()]
    extracted = [r for r in have if (PROC / "landmarks" / f"{r['video_id']}.npz").exists()]
    return {"videos_listed": len(rows), "videos_downloaded": len(have), "videos_extracted": len(extracted),
            "glosses": len({r["gloss"] for r in rows}), "signers": len({r["signer_id"] for r in have}),
            "split": {s: sum(r["split"] == s for r in have) for s in ("train", "val", "test")}}


def dashboard():
    tab = read(RESULTS / "tabular_results.json")
    cv = read(RESULTS / "tabular_app20_cv.json")
    nlp = read(RESULTS / "nlp_results.json")
    tmpl = read(RESULTS / "template_recognizer_results.json")
    if nlp:
        nlp = dict(nlp)
        nlp["app_examples"] = nlp.get("app_examples", [])[:40]
    return {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "wlasl": wlasl_facts(),
        "cleaning": read(PROC / "cleaning_report.json"),
        "pdf": read(PROC / "pdf" / "cleaning_report.json"),
        "tabular": tab,
        "tabular_app20_cv": drop(cv, "pooled_predictions"),
        "template_recognizer": drop(tmpl, "rows"),
        "images": read(RESULTS / "image_results.json"),
        "images_random_validation": read(RESULTS / "image_results_random_validation.json"),
        "nlp": nlp,
        "pose": read(RESULTS / "pose_results.json"),
        "figures": sorted(p.name for p in (ROOT / "figures").glob("*.png")),
    }


if __name__ == "__main__":
    DEST.mkdir(parents=True, exist_ok=True)
    for src, name in FILES.items():
        if src.exists():
            shutil.copyfile(src, DEST / name)
            print(f"{name:28s} {src.stat().st_size / 1024:8.0f} KB")
        else:
            print(f"{name:28s} missing ({src.name} has not been made yet)")
    (DEST / "figures").mkdir(exist_ok=True)
    for png in (ROOT / "figures").glob("*.png"):
        shutil.copyfile(png, DEST / "figures" / png.name)
    data = dashboard()
    (DEST / "dashboard.json").write_text(json.dumps(data))
    print(f"dashboard.json with {sum(v is not None for v in data.values())} parts, {len(data['figures'])} figures")
