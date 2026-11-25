"""Sign recognition from the landmark table.

Input: data/processed/wlasl_features.csv.gz (one row per frame, from clean_landmarks.py)
and data/processed/sequences.npz (the same rows grouped by video).

Tasks
  wlasl100  the 100 most common WLASL glosses (the standard WLASL100 benchmark)
  app20     the 20 words the app's camera recognises (src/camera/core-words.js)

Split: WLASL's official train / val / test split, by video, so frames of one
video are never on both sides. Signers are not separated by WLASL's split; how
many test videos come from signers also seen in training is reported.

Models, each scored per video (top-1 and top-5):
  1. Random Forest, one prediction per frame, averaged over the video (baseline)
  2. XGBoost, same frame-level setup
  3. XGBoost on one row per video: the signing part of the video resampled to 8 steps, plus summary statistics
  4. GRU sequence model on the signing part resampled to 32 steps, with augmentation
  5. Ensemble of 3 and 4 (mean of probabilities)

Class imbalance (4-16 videos per sign): every model is trained with balanced class
weights (balanced_weights); model 3 is also trained without them, for comparison.
Macro-F1 is reported next to accuracy because it weighs every sign equally.

Outputs: results/tabular_results.json, figures/tabular_*.png, models/signs_*.onnx
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
MODELS = ROOT / "models"
SEED = 361
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)

APP20 = ["hello", "thank you", "please", "sorry", "yes", "no", "help", "water", "eat", "sad", "more",
         "name", "friend", "doctor", "go", "want", "sick", "tired", "understand", "again"]
STEPS_XGB, STEPS_SEQ = 8, 32


# ── data ─────────────────────────────────────────────────────────────────────

def load():
    names = json.loads((PROC / "feature_names.json").read_text())
    videos = pd.read_csv(PROC / "videos.csv", dtype={"video_id": str})
    seqs = np.load(PROC / "sequences.npz")
    return names, videos, {v: seqs[v] for v in videos["video_id"]}


def active_part(x, names):
    """The signing part of a video: frames where the dominant hand is seen or raised."""
    i_y, i_p = names.index("dom_y"), names.index("dom_present")
    act = (x[:, i_p] > 0.5) | (np.nan_to_num(x[:, i_y], nan=-9) > -1.0)
    idx = np.where(act)[0]
    if len(idx) < 4:
        return x
    return x[idx[0]:idx[-1] + 1]


def resample(x, steps):
    """Linear resampling in time, per feature, ignoring missing values."""
    n, d = x.shape
    t_old = np.linspace(0, 1, n)
    t_new = np.linspace(0, 1, steps)
    out = np.full((steps, d), np.nan, dtype=np.float32)
    for j in range(d):
        ok = ~np.isnan(x[:, j])
        if ok.sum() >= 2:
            out[:, j] = np.interp(t_new, t_old[ok], x[ok, j])
        elif ok.sum() == 1:
            out[:, j] = x[ok, j][0]
    return out


COMPACT = ["dom_x", "dom_y", "dom_tip_x", "dom_tip_y", "oth_x", "oth_y", "dom_present", "oth_present",
           "dom_elbow_x", "dom_elbow_y"] + [f"dom_shape{k}" for k in range(63, 78)] + [f"oth_shape{k}" for k in range(63, 78)]


def video_row(x, names):
    """One row per video for model 3."""
    a = active_part(x, names)
    cols = [names.index(c) for c in COMPACT]
    r = resample(a[:, cols], STEPS_XGB).ravel()
    pick = lambda c: a[:, names.index(c)]
    stats = []
    for c in ("dom_x", "dom_y", "oth_x", "oth_y", "dom_tip_x", "dom_tip_y"):
        v = pick(c)
        stats += [np.nanmean(v), np.nanstd(v), np.nanmin(v), np.nanmax(v)] if (~np.isnan(v)).any() else [np.nan] * 4
    speed = np.hypot(pick("dom_vx"), pick("dom_vy"))
    stats += [len(a) / 25.0, float(np.nansum(speed) / 25.0), float(np.nanmax(speed)), float(np.nanmean(pick("oth_present")))]
    return np.concatenate([r, np.array(stats, dtype=np.float32)])


def video_row_names():
    n = [f"t{t}_{c}" for t in range(STEPS_XGB) for c in COMPACT]
    for c in ("dom_x", "dom_y", "oth_x", "oth_y", "dom_tip_x", "dom_tip_y"):
        n += [f"{c}_mean", f"{c}_std", f"{c}_min", f"{c}_max"]
    return n + ["duration_s", "dom_path", "dom_peak_speed", "oth_visible_share"]


# ── models ───────────────────────────────────────────────────────────────────

def balanced_weights(y):
    """Class imbalance: every class gets the same total weight (inverse class frequency, mean weight 1).

    WLASL has 4-16 videos per sign, and long videos have more frames than short ones,
    so without this the common and the slowly signed words would dominate training.
    """
    cls, n = np.unique(y, return_counts=True)
    w = dict(zip(cls, len(y) / (len(cls) * n)))
    return np.array([w[v] for v in y], dtype=np.float32)


def frame_models(Xtr, ytr, Xte, vte, ncls):
    """Models 1 and 2: per-frame classifiers, averaged per video (both class-balanced)."""
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    out = {}
    w = balanced_weights(ytr)
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt", n_jobs=4, random_state=SEED)
    rf.fit(np.nan_to_num(Xtr), ytr, sample_weight=w)
    out["rf"] = (rf.predict_proba(np.nan_to_num(Xte)), rf)
    xgb = XGBClassifier(n_estimators=250, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.5,
                        tree_method="hist", n_jobs=4, random_state=SEED)
    xgb.fit(Xtr, ytr, sample_weight=w)
    out["xgb_frame"] = (xgb.predict_proba(Xte), xgb)
    res = {}
    for k, (p, m) in out.items():
        full = np.zeros((len(p), ncls))
        full[:, m.classes_] = p
        res[k] = (by_video(full, vte), m)
    return res


def by_video(p, vids):
    df = pd.DataFrame(p)
    df["v"] = vids
    g = df.groupby("v", sort=False).mean()
    return g.index.to_numpy(), g.to_numpy()


class GRUNet(nn.Module):
    def __init__(self, d, ncls, h=128):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.2))
        self.gru = nn.GRU(h, h, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.att = nn.Linear(2 * h, 1)
        self.out = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * h, ncls))

    def forward(self, x):
        z, _ = self.gru(self.inp(x))
        w = torch.softmax(self.att(z), 1)
        return self.out((w * z).sum(1))


def seq_tensor(seqs, ids, names, mean=None, std=None):
    X = np.stack([resample(active_part(seqs[v], names), STEPS_SEQ) for v in ids])
    present = ~np.isnan(X)
    if mean is None:
        mean = np.nanmean(X.reshape(-1, X.shape[-1]), 0)
        std = np.nanstd(X.reshape(-1, X.shape[-1]), 0) + 1e-3
        mean, std = np.nan_to_num(mean), np.nan_to_num(std, nan=1.0)
    X = np.where(present, (X - mean) / std, 0.0).astype(np.float32)
    return X, mean, std


POSITION_COLS = ("dom_x", "dom_y", "dom_tip_x", "dom_tip_y", "oth_x", "oth_y", "oth_tip_x", "oth_tip_y",
                 "dom_elbow_x", "dom_elbow_y", "oth_elbow_x", "oth_elbow_y", "nose_x", "nose_y")


def augment(xb, names, mean, std):
    """Random body size, position, a small tilt, timing changes and noise (on normalised features)."""
    b, t, d = xb.shape
    x = xb.clone()
    pos = [names.index(c) for c in POSITION_COLS]
    xs, ys = pos[0::2], pos[1::2]
    m = torch.tensor(mean, dtype=torch.float32)
    s = torch.tensor(std, dtype=torch.float32)
    raw = x * s + m
    scale = torch.empty(b, 1, 1).uniform_(0.88, 1.12)
    ang = torch.empty(b, 1).uniform_(-0.15, 0.15)
    shift = torch.empty(b, 1, 2).uniform_(-0.08, 0.08)
    px, py = raw[:, :, xs], raw[:, :, ys]
    c, sn = torch.cos(ang)[:, :, None], torch.sin(ang)[:, :, None]
    raw[:, :, xs] = (c * px - sn * py) * scale + shift[:, :, :1]
    raw[:, :, ys] = (sn * px + c * py) * scale + shift[:, :, 1:]
    x = (raw - m) / s
    x = torch.where(xb == 0, xb, x)  # missing values (stored as 0) stay missing
    # Timing: resample from a random 80-100% window.
    out = torch.empty_like(x)
    for i in range(b):
        L = int(t * np.random.uniform(0.8, 1.0))
        st = np.random.randint(0, t - L + 1)
        idx = torch.linspace(st, st + L - 1, t).round().long()
        out[i] = x[i, idx]
    return out + 0.03 * torch.randn_like(out)


def train_gru(Xtr, ytr, Xva, yva, ncls, names, mean, std, epochs=120, log=None):
    model = GRUNet(Xtr.shape[-1], ncls)
    counts = np.bincount(ytr, minlength=ncls).astype(np.float32)
    class_w = torch.tensor(np.where(counts > 0, len(ytr) / (ncls * np.maximum(counts, 1)), 1.0), dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * int(np.ceil(len(Xtr) / 32)))
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    Xv = torch.tensor(Xva)
    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(Xt), 32):
            idx = perm[i:i + 32]
            xb = augment(Xt[idx], names, mean, std)
            loss = nn.functional.cross_entropy(model(xb), yt[idx], weight=class_w, label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            acc = float((model(Xv).argmax(1).numpy() == yva).mean())
        if log is not None:
            log.append({"epoch": ep + 1, "train_loss": tot / len(Xt), "val_acc": acc})
        if acc >= best and ep >= epochs // 4:
            best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model


# ── evaluation ───────────────────────────────────────────────────────────────

def topk(p, y, k):
    return float(np.mean([y[i] in np.argsort(-p[i])[:k] for i in range(len(y))]))


def score(p, y):
    from sklearn.metrics import f1_score
    return {"top1": topk(p, y, 1), "top5": topk(p, y, 5), "macro_f1": float(f1_score(y, p.argmax(1), average="macro")), "videos": int(len(y))}


def run_task(task, names, videos, seqs, out):
    t0 = time.time()
    if task == "wlasl100":
        sel = videos[videos["gloss_rank"] < 100]
    else:
        sel = videos[videos["gloss"].isin(APP20)]
    classes = sorted(sel["gloss"].unique())
    cid = {g: i for i, g in enumerate(classes)}
    tr, va, te = (sel[sel["split"] == s] for s in ("train", "val", "test"))
    ncls = len(classes)
    y = {s: d["gloss"].map(cid).to_numpy() for s, d in (("train", tr), ("val", va), ("test", te))}
    res = {"classes": ncls, "videos": {"train": len(tr), "val": len(va), "test": len(te)},
           "test_videos_from_signers_seen_in_training": float(te["signer_id"].isin(set(tr["signer_id"]) | set(va["signer_id"])).mean())}

    # Frame tables for models 1-2 (train + val frames are used for training: these models have no early stopping).
    def frames(d):
        X = np.concatenate([seqs[v] for v in d["video_id"]])
        lab = np.concatenate([[cid[g]] * len(seqs[v]) for v, g in zip(d["video_id"], d["gloss"])])
        vid = np.concatenate([[v] * len(seqs[v]) for v in d["video_id"]])
        return X, lab, vid
    trva = pd.concat([tr, va])
    Xf, yf, _ = frames(trva)
    Xft, yft, vft = frames(te)
    res["frame_rows"] = {"train": int(len(Xf)), "test": int(len(Xft))}
    order = te["video_id"].to_numpy()
    fm = frame_models(Xf, yf, Xft, vft, ncls)
    for k, ((vids, p), m) in fm.items():
        p = pd.DataFrame(p, index=vids).loc[order].to_numpy()
        res[k] = score(p, y["test"])
        res[k]["frame_top1"] = float((m.predict(np.nan_to_num(Xft) if k == "rf" else Xft) == yft).mean()) if hasattr(m, "predict") else None
        if k == "xgb_frame":
            p_frame = p
    print(task, "frame models", {k: res[k]["top1"] for k in fm}, f"{time.time() - t0:.0f}s", flush=True)

    # Model 3: one row per video.
    from xgboost import XGBClassifier
    V = {s: np.stack([video_row(seqs[v], names) for v in d["video_id"]]) for s, d in (("train", tr), ("val", va), ("test", te))}
    if len(np.unique(y["train"])) == ncls:
        # Number of trees by early stopping on the validation videos.
        xgb_v = XGBClassifier(n_estimators=600, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.4,
                              min_child_weight=1, tree_method="hist", n_jobs=4, random_state=SEED, early_stopping_rounds=60,
                              eval_metric="mlogloss")
        xgb_v.fit(V["train"], y["train"], sample_weight=balanced_weights(y["train"]), eval_set=[(V["val"], y["val"])], verbose=False)
        best_n = int(xgb_v.best_iteration) + 1
    else:
        best_n = 300  # a sign has no training video of its own in WLASL's split: no early stopping possible
    y_trva = np.concatenate([y["train"], y["val"]])
    counts = np.bincount(y_trva, minlength=ncls)
    res["class_balance"] = {"videos_per_class": {classes[c]: int(n) for c, n in enumerate(counts)},
                            "min": int(counts.min()), "max": int(counts.max()),
                            "imbalance_ratio": round(float(counts.max() / max(1, counts.min())), 2)}
    V_trva = np.concatenate([V["train"], V["val"]])

    def video_xgb(weighted):
        m = XGBClassifier(n_estimators=best_n, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.4,
                          tree_method="hist", n_jobs=4, random_state=SEED)
        m.fit(V_trva, y_trva, sample_weight=balanced_weights(y_trva) if weighted else None)
        p = np.zeros((len(te), ncls))
        p[:, m.classes_] = m.predict_proba(V["test"])
        return m, p

    xgb_v2, p3 = video_xgb(weighted=True)
    res["xgb_video"] = score(p3, y["test"])
    res["xgb_video"]["trees"] = best_n
    # The same model without class weights, to show what balancing changes.
    res["xgb_video_unweighted"] = score(video_xgb(weighted=False)[1], y["test"])
    imp = pd.Series(xgb_v2.feature_importances_, index=video_row_names()).sort_values(ascending=False)
    res["xgb_video_top_features"] = {k: float(v) for k, v in imp.head(15).items()}
    print(task, "video xgb", res["xgb_video"]["top1"], flush=True)

    # Model 4: GRU.
    Xs_tr, mean, std = seq_tensor(seqs, tr["video_id"], names)
    Xs_va, _, _ = seq_tensor(seqs, va["video_id"], names, mean, std)
    Xs_te, _, _ = seq_tensor(seqs, te["video_id"], names, mean, std)
    log = []
    gru = train_gru(Xs_tr, y["train"], Xs_va, y["val"], ncls, names, mean, std, log=log)
    with torch.no_grad():
        p4 = torch.softmax(gru(torch.tensor(Xs_te)), 1).numpy()
    res["gru"] = score(p4, y["test"])
    res["gru_curve"] = log
    res["ensemble"] = score((p3 + p4) / 2, y["test"])
    print(task, "gru", res["gru"]["top1"], "ensemble", res["ensemble"]["top1"], flush=True)

    res["seconds"] = round(time.time() - t0)
    out[task] = res
    preds = {"classes": classes, "y": y["test"].tolist(), "xgb_frame": p_frame.tolist(), "xgb_video": p3.tolist(),
             "gru": p4.tolist(), "video_ids": order.tolist()}
    (RESULTS / f"tabular_{task}_predictions.json").write_text(json.dumps(preds))
    if task == "app20":
        export_app_models(gru, xgb_v2, names, classes, mean, std, V, y)
    return res


def export_app_models(gru, xgb_v, names, classes, mean, std, V, y):
    """The app20 models, for the browser (onnxruntime-web)."""
    MODELS.mkdir(exist_ok=True)
    torch.onnx.export(gru, torch.zeros(1, STEPS_SEQ, len(names)), MODELS / "signs_app20_gru.onnx", input_names=["seq"],
                      output_names=["logits"], dynamic_axes={"seq": {0: "n"}}, opset_version=17, dynamo=False)
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType
    onx = convert_xgboost(xgb_v, initial_types=[("row", FloatTensorType([None, V["train"].shape[1]]))], target_opset=15)
    (MODELS / "signs_app20_xgb.onnx").write_bytes(onx.SerializeToString())
    meta = {"classes": classes, "feature_names": names, "seq_steps": STEPS_SEQ, "seq_mean": mean.tolist(),
            "seq_std": std.tolist(), "video_row_steps": STEPS_XGB, "video_row_compact": COMPACT,
            "video_row_names": video_row_names()}
    (MODELS / "signs_app20_meta.json").write_text(json.dumps(meta))


def plots(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"rf": "Random Forest\n(frames, baseline)", "xgb_frame": "XGBoost\n(frames)", "xgb_video": "XGBoost\n(per video)",
              "gru": "GRU\n(sequence)", "ensemble": "Ensemble\n(XGB video + GRU)"}
    tasks = [t for t in ("wlasl100", "app20") if t in out]
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.2 * len(tasks), 3.8), squeeze=False)
    for ax, t in zip(axes[0], tasks):
        ks = list(labels)
        t1 = [out[t][k]["top1"] * 100 for k in ks]
        t5 = [out[t][k]["top5"] * 100 for k in ks]
        x = np.arange(len(ks))
        ax.bar(x - 0.2, t1, 0.4, label="Top-1", color="#2f6fed")
        ax.bar(x + 0.2, t5, 0.4, label="Top-5", color="#9db8f5")
        for xi, v in zip(x, t1):
            ax.text(xi - 0.2, v + 1, f"{v:.0f}", ha="center", fontsize=8)
        ax.axhline(70, color="#c0392b", lw=1, ls="--")
        ax.set_xticks(x, [labels[k] for k in ks], fontsize=7)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Test accuracy per video (%)")
        n = out[t]["classes"]
        ax.set_title(f"{'WLASL100' if t == 'wlasl100' else 'App camera words'} ({n} signs, {out[t]['videos']['test']} test videos)", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "tabular_accuracy.png", dpi=160)
    plt.close(fig)

    if "app20" in out:
        pr = json.loads((RESULTS / "tabular_app20_predictions.json").read_text())
        p = (np.array(pr["xgb_video"]) + np.array(pr["gru"])) / 2
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(pr["y"], p.argmax(1), labels=range(len(pr["classes"])))
        fig, ax = plt.subplots(figsize=(6.6, 5.8))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(pr["classes"])), pr["classes"], rotation=70, fontsize=7)
        ax.set_yticks(range(len(pr["classes"])), pr["classes"], fontsize=7)
        for i in range(len(cm)):
            for j in range(len(cm)):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("App camera words, ensemble, test videos", fontsize=10)
        fig.tight_layout()
        fig.savefig(FIGS / "tabular_confusion_app20.png", dpi=160)
        plt.close(fig)

    for t in tasks:
        cb = out[t].get("class_balance")
        if cb:
            items = sorted(cb["videos_per_class"].items(), key=lambda kv: -kv[1])
            fig, ax = plt.subplots(figsize=(9 if t == "app20" else 12, 3.0))
            ax.bar(range(len(items)), [n for _, n in items], color="#2f6fed")
            if len(items) <= 30:
                ax.set_xticks(range(len(items)), [g for g, _ in items], rotation=70, fontsize=7)
            else:
                ax.set_xticks([])
                ax.set_xlabel("signs, most to fewest training videos")
            ax.set_ylabel("training videos")
            ax.set_title(f"Class balance ({t}): {cb['min']}-{cb['max']} videos per sign, ratio {cb['imbalance_ratio']}", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            fig.savefig(FIGS / f"tabular_class_balance_{t}.png", dpi=160)
            plt.close(fig)
        imp = out[t]["xgb_video_top_features"]
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ax.barh(list(imp)[::-1], list(imp.values())[::-1], color="#2f6fed")
        ax.set_title(f"Most used features, per-video XGBoost ({t})", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIGS / f"tabular_importance_{t}.png", dpi=160)
        plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["wlasl100", "app20", "all"], default="all")
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    names, videos, seqs = load()
    path = RESULTS / "tabular_results.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    out["feature_columns"] = len(names)
    tasks = ("wlasl100", "app20") if a.task == "all" else (a.task,)
    for task in tasks:
        run_task(task, names, videos, seqs, out)
        path.write_text(json.dumps(out, indent=1))
    plots(out)
    print(json.dumps({t: {k: out[t][k]["top1"] for k in ("rf", "xgb_frame", "xgb_video", "gru", "ensemble")}
                      for t in tasks}, indent=1))


if __name__ == "__main__":
    main()
