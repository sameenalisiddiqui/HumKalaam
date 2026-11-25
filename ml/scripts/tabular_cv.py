"""5-fold cross-validation for the app's 20 camera words.

WLASL's official split leaves only about one test video per word for these 20
words, so a single test score moves by 5 points per video. Here every app20
video is used for testing exactly once: stratified 5-fold cross-validation by
video (folds are made of whole videos, so no frames of a test video are trained
on). Inside each fold, 15% of the training videos are held out for the GRU's and
XGBoost's early stopping. Models and settings are those of tabular_models.py,
with balanced class weights.

Output: results/tabular_app20_cv.json, figures/tabular_app20_cv.png
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from tabular_models import (APP20, FIGS, RESULTS, SEED, balanced_weights, frame_models, load, score,  # noqa: E402
                            seq_tensor, train_gru, video_row)


def main():
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from xgboost import XGBClassifier
    t0 = time.time()
    names, videos, seqs = load()
    sel = videos[videos["gloss"].isin(APP20)].reset_index(drop=True)
    classes = sorted(sel["gloss"].unique())
    cid = {g: i for i, g in enumerate(classes)}
    y_all = sel["gloss"].map(cid).to_numpy()
    ncls = len(classes)
    V_all = np.stack([video_row(seqs[v], names) for v in sel["video_id"]])
    per = {k: [] for k in ("rf", "xgb_frame", "xgb_video", "gru", "ensemble")}
    pooled = {k: np.zeros((len(sel), ncls)) for k in per}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (tr_i, te_i) in enumerate(skf.split(V_all, y_all)):
        tr_i, va_i = train_test_split(tr_i, test_size=0.15, stratify=y_all[tr_i], random_state=SEED)
        tr, va, te = sel.iloc[tr_i], sel.iloc[va_i], sel.iloc[te_i]
        yte = y_all[te_i]

        # Frame models, trained on train + validation videos.
        trva = pd.concat([tr, va])
        Xf = np.concatenate([seqs[v] for v in trva["video_id"]])
        yf = np.concatenate([[cid[g]] * len(seqs[v]) for v, g in zip(trva["video_id"], trva["gloss"])])
        Xt = np.concatenate([seqs[v] for v in te["video_id"]])
        vt = np.concatenate([[v] * len(seqs[v]) for v in te["video_id"]])
        for k, ((vids, p), _) in frame_models(Xf, yf, Xt, vt, ncls).items():
            p = pd.DataFrame(p, index=vids).loc[te["video_id"].to_numpy()].to_numpy()
            per[k].append(score(p, yte))
            pooled[k][te_i] = p

        # Per-video XGBoost, trees by early stopping on the fold's validation videos.
        m = XGBClassifier(n_estimators=600, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.4,
                          tree_method="hist", n_jobs=4, random_state=SEED, early_stopping_rounds=60, eval_metric="mlogloss")
        m.fit(V_all[tr_i], y_all[tr_i], sample_weight=balanced_weights(y_all[tr_i]), eval_set=[(V_all[va_i], y_all[va_i])], verbose=False)
        n_trees = int(m.best_iteration) + 1
        m = XGBClassifier(n_estimators=n_trees, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.4,
                          tree_method="hist", n_jobs=4, random_state=SEED)
        idx = np.concatenate([tr_i, va_i])
        m.fit(V_all[idx], y_all[idx], sample_weight=balanced_weights(y_all[idx]))
        p3 = np.zeros((len(te_i), ncls))
        p3[:, m.classes_] = m.predict_proba(V_all[te_i])
        per["xgb_video"].append(score(p3, yte))
        pooled["xgb_video"][te_i] = p3

        # GRU.
        Xs_tr, mean, std = seq_tensor(seqs, tr["video_id"], names)
        Xs_va, _, _ = seq_tensor(seqs, va["video_id"], names, mean, std)
        Xs_te, _, _ = seq_tensor(seqs, te["video_id"], names, mean, std)
        gru = train_gru(Xs_tr, y_all[tr_i], Xs_va, y_all[va_i], ncls, names, mean, std)
        with torch.no_grad():
            p4 = torch.softmax(gru(torch.tensor(Xs_te)), 1).numpy()
        per["gru"].append(score(p4, yte))
        pooled["gru"][te_i] = p4
        per["ensemble"].append(score((p3 + p4) / 2, yte))
        pooled["ensemble"][te_i] = (p3 + p4) / 2
        print(f"fold {fold + 1}: " + ", ".join(f"{k} {per[k][-1]['top1']:.2f}" for k in per), f"{time.time() - t0:.0f}s", flush=True)

    out = {"videos": int(len(sel)), "classes": ncls, "folds": 5}
    for k, folds in per.items():
        t1 = np.array([f["top1"] for f in folds])
        t5 = np.array([f["top5"] for f in folds])
        f1 = np.array([f["macro_f1"] for f in folds])
        out[k] = {"top1_mean": float(t1.mean()), "top1_sd": float(t1.std(ddof=1)), "top5_mean": float(t5.mean()),
                  "macro_f1_mean": float(f1.mean()), "pooled": score(pooled[k], y_all)}
    out["pooled_predictions"] = {"classes": classes, "y": y_all.tolist(), "ensemble": pooled["ensemble"].tolist(),
                                 "rf": pooled["rf"].tolist()}
    out["seconds"] = round(time.time() - t0)
    (RESULTS / "tabular_app20_cv.json").write_text(json.dumps(out, indent=1))
    plot(out, classes, y_all, pooled)
    print(json.dumps({k: {"top1": round(out[k]["top1_mean"], 3), "sd": round(out[k]["top1_sd"], 3),
                          "top5": round(out[k]["top5_mean"], 3)} for k in per}, indent=1))


def plot(out, classes, y, pooled):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    labels = {"rf": "Random Forest\n(frames, baseline)", "xgb_frame": "XGBoost\n(frames)", "xgb_video": "XGBoost\n(per video)",
              "gru": "GRU\n(sequence)", "ensemble": "Ensemble\n(XGB video + GRU)"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    ks = list(labels)
    m = [out[k]["top1_mean"] * 100 for k in ks]
    sd = [out[k]["top1_sd"] * 100 for k in ks]
    t5 = [out[k]["top5_mean"] * 100 for k in ks]
    x = np.arange(len(ks))
    ax.bar(x - 0.2, m, 0.4, yerr=sd, capsize=3, label="Top-1 (mean ± sd over 5 folds)", color="#2f6fed")
    ax.bar(x + 0.2, t5, 0.4, label="Top-5", color="#9db8f5")
    for xi, v in zip(x, m):
        ax.text(xi - 0.2, v + 2, f"{v:.0f}", ha="center", fontsize=8)
    ax.axhline(70, color="#c0392b", lw=1, ls="--")
    ax.set_xticks(x, [labels[k] for k in ks], fontsize=7)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy per video (%)")
    ax.set_title(f"App's 20 camera words, 5-fold CV ({out['videos']} WLASL videos)", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    best = max(ks, key=lambda k: out[k]["top1_mean"])
    cm = confusion_matrix(y, pooled[best].argmax(1), labels=range(len(classes)))
    ax = axes[1]
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=70, fontsize=7)
    ax.set_yticks(range(len(classes)), classes, fontsize=7)
    for i in range(len(cm)):
        for j in range(len(cm)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix, {labels[best].replace(chr(10), ' ')}, all folds", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "tabular_app20_cv.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
