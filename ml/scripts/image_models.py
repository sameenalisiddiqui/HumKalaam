"""ASL letters from images, and the image + landmark fusion.

Data (download_data.py): 500 labelled training images, 100 unlabelled images,
260 held-out test images (a later stretch of each letter's recording) and
Kaggle's own 26 test photos, all from grassknoted/asl-alphabet.

Models
  1. Baseline        logistic regression on 32x32 grey pixels
  2. Linear probe    EfficientNet-B0 (ImageNet) features, frozen, + logistic regression
  3. Fine-tuned      EfficientNet-B0, last blocks + classifier trained, with augmentation
  4. Semi-supervised 3, then pseudo-label the 100 unlabelled images (kept only when the
                     model is at least 90% sure) and train again on both
  5. Landmarks       MediaPipe hand landmarks of the same images -> XGBoost
  6. Fusion          weighted average of 4 and 5 (weight chosen on validation);
                     the CNN alone when MediaPipe finds no hand

The labels of the unlabelled images are read only after training, to report how
many pseudo-labels were right.

Outputs: results/image_results.json, figures/image_*.png, models/letters_cnn.onnx,
models/letters_landmarks.json
"""
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT.parent
DATA = ROOT / "data" / "raw" / "asl_alphabet"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
MODELS = ROOT / "models"
LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
SEED = 361
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)


def load_manifest():
    with open(DATA / "manifest.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by = {}
    for r in rows:
        by.setdefault(r["use"], []).append((DATA / r["file"], LETTERS.index(r["letter"])))
    return by


def frame_number(path):
    return int("".join(c for c in Path(path).stem if c.isdigit()))


def split_labelled(items, cut=1200):
    """Train / validation split of the 500 labelled images BY STRETCH of each recording.

    The labelled images come from frames 1-1500 of each letter's recording. Frames
    above `cut` (about a fifth) are the validation set, so validation images are not
    near-copies of training images. (A random split gave 100% validation accuracy,
    which could not tell models or fusion weights apart.)
    """
    train = [(p, y) for p, y in items if frame_number(p) <= cut]
    val = [(p, y) for p, y in items if frame_number(p) > cut]
    return train, val


# ── pixels baseline ──────────────────────────────────────────────────────────

def pixels(items, size=32):
    return np.stack([np.asarray(Image.open(p).convert("L").resize((size, size)), dtype=np.float32).ravel() / 255.0
                     for p, _ in items])


# ── EfficientNet ─────────────────────────────────────────────────────────────

def tfms(train):
    from torchvision import transforms as T
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if train:
        return T.Compose([T.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.9, 1.1)), T.RandomRotation(12),
                          T.ColorJitter(0.35, 0.35, 0.25, 0.03), T.ToTensor(), norm])
    return T.Compose([T.Resize(224), T.ToTensor(), norm])


class Images(torch.utils.data.Dataset):
    def __init__(self, items, train):
        self.items, self.t = items, tfms(train)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.t(Image.open(p).convert("RGB")), y


def effnet():
    import torchvision
    m = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(LETTERS))
    return m


@torch.no_grad()
def predict(model, items, batch=32):
    model.eval()
    dl = torch.utils.data.DataLoader(Images(items, False), batch_size=batch)
    return torch.cat([torch.softmax(model(x), 1) for x, _ in dl]).numpy()


@torch.no_grad()
def features(items):
    m = effnet()
    m.classifier = nn.Identity()
    m.eval()
    dl = torch.utils.data.DataLoader(Images(items, False), batch_size=32)
    return torch.cat([m(x) for x, _ in dl]).numpy()


def train_cnn(train, val, epochs=12, log=None):
    """Fine-tune the last three feature blocks and the classifier."""
    m = effnet()
    for p in m.parameters():
        p.requires_grad = False
    for block in list(m.features)[-3:]:
        for p in block.parameters():
            p.requires_grad = True
    for p in m.classifier.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    dl = torch.utils.data.DataLoader(Images(train, True), batch_size=32, shuffle=True,
                                     generator=torch.Generator().manual_seed(SEED))
    best, best_state = -1, None
    for ep in range(epochs):
        m.train()
        tot, n = 0.0, 0
        for x, y in dl:
            loss = nn.functional.cross_entropy(m(x), y, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            n += len(y)
        sched.step()
        acc = float((predict(m, val).argmax(1) == np.array([y for _, y in val])).mean())
        if log is not None:
            log.append({"epoch": ep + 1, "train_loss": tot / n, "val_acc": acc})
        print(f"  epoch {ep + 1}: loss {tot / n:.3f} val {acc:.3f}", flush=True)
        if acc >= best:
            best, best_state = acc, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state)
    return m


# ── hand landmarks ───────────────────────────────────────────────────────────

# MediaPipe finds a hand in few of these small, dark, edge-cropped images. Measured on
# half the test images: 37% at confidence 0.3, 38% at 0.2, 45% at 0.1; padding or
# upscaling the image did not help (25-38%). The lowest threshold is used, and the
# fusion weight (chosen on validation) decides how far to trust what it finds.
HAND_CONFIDENCE = 0.1


def landmarks(items):
    """MediaPipe hand world landmarks (21 x 3) per image, the app's model; NaN when no hand is found."""
    import mediapipe as mp
    v = mp.tasks.vision
    det = v.HandLandmarker.create_from_options(v.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(APP / "public" / "models" / "hand_landmarker.task")),
        running_mode=v.RunningMode.IMAGE, num_hands=1, min_hand_detection_confidence=HAND_CONFIDENCE,
        min_hand_presence_confidence=HAND_CONFIDENCE))
    out = np.full((len(items), 21, 3), np.nan, dtype=np.float32)
    for i, (p, _) in enumerate(items):
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(Image.open(p).convert("RGB")))
        r = det.detect(img)
        if r.hand_world_landmarks:
            out[i] = [(q.x, q.y, q.z) for q in r.hand_world_landmarks[0]]
    det.close()
    return out


def hand_features(world):
    """Handshape features that do not depend on where the hand is or how big it is.

    Points relative to the wrist, scaled by the wrist-to-middle-knuckle length,
    plus the pairwise distances between the fingertips and the wrist.
    The same function is mirrored in the app (src/camera/learned-features.js).
    """
    w = world - world[:, :1]
    scale = np.linalg.norm(w[:, 9], axis=1, keepdims=True)[:, :, None]
    w = w / np.where(scale > 1e-6, scale, 1)
    tips = [0, 4, 8, 12, 16, 20]
    d = np.stack([np.linalg.norm(w[:, a] - w[:, b], axis=1) for i, a in enumerate(tips) for b in tips[i + 1:]], 1)
    return np.concatenate([w.reshape(len(w), -1), d], 1)


# ── metrics and plots ────────────────────────────────────────────────────────

def scores(y, p):
    from sklearn.metrics import accuracy_score, f1_score
    return {"accuracy": float(accuracy_score(y, p)), "macro_f1": float(f1_score(y, p, average="macro")), "n": int(len(y))}


def confusion_plot(y, p, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, p, labels=range(26))
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(26), LETTERS, fontsize=7)
    ax.set_yticks(range(26), LETTERS, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    for i in range(26):
        for j in range(26):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6, color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from xgboost import XGBClassifier

    RESULTS.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    data = load_manifest()
    train, val = split_labelled(data["labelled"])
    test, ktest = data["test"], data["kaggle_test"]
    unl = [(p, -1) for p, _ in data["unlabelled"]]  # labels hidden
    unl_truth = np.array([y for _, y in data["unlabelled"]])  # read only for scoring pseudo-labels
    Y = {k: np.array([y for _, y in v]) for k, v in (("train", train), ("val", val), ("test", test), ("ktest", ktest))}
    out = {"sizes": {"train": len(train), "val": len(val), "unlabelled": len(unl), "test": len(test), "kaggle_test": len(ktest)}}
    t0 = time.time()

    # 1. pixels baseline
    base = make_pipeline(StandardScaler(), PCA(60, random_state=SEED), LogisticRegression(max_iter=3000, C=0.5))
    base.fit(pixels(train), Y["train"])
    out["1_pixels_logreg"] = {s: scores(Y[s], base.predict(pixels(items))) for s, items in (("test", test), ("ktest", ktest))}
    print("baseline", out["1_pixels_logreg"], flush=True)

    # 2. frozen EfficientNet + logistic regression
    F = {k: features(v) for k, v in (("train", train), ("test", test), ("ktest", ktest))}
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5)).fit(F["train"], Y["train"])
    out["2_effnet_linear_probe"] = {s: scores(Y[s], probe.predict(F[s])) for s in ("test", "ktest")}
    print("probe", out["2_effnet_linear_probe"], flush=True)

    # 3. fine-tuned EfficientNet
    log3 = []
    cnn = train_cnn(train, val, log=log3)
    P3 = {k: predict(cnn, v) for k, v in (("val", val), ("test", test), ("ktest", ktest))}
    out["3_effnet_finetuned"] = {s: scores(Y[s], P3[s].argmax(1)) for s in ("test", "ktest")}
    out["3_curve"] = log3
    print("finetuned", out["3_effnet_finetuned"], flush=True)

    # 4. pseudo-labels on the unlabelled images, then train again
    pu = predict(cnn, unl)
    keep = pu.max(1) >= 0.90
    pseudo = [(p, int(c)) for (p, _), c, k in zip(unl, pu.argmax(1), keep) if k]
    out["4_pseudo_labels"] = {"kept": int(keep.sum()), "of": len(unl),
                              "correct_among_kept": float((pu.argmax(1)[keep] == unl_truth[keep]).mean()) if keep.any() else None}
    log4 = []
    cnn2 = train_cnn(train + pseudo, val, log=log4)
    P4 = {k: predict(cnn2, v) for k, v in (("val", val), ("test", test), ("ktest", ktest))}
    out["4_effnet_semi_supervised"] = {s: scores(Y[s], P4[s].argmax(1)) for s in ("test", "ktest")}
    out["4_curve"] = log4
    print("semi", out["4_pseudo_labels"], out["4_effnet_semi_supervised"], flush=True)

    # 5. landmarks
    L = {k: landmarks(v) for k, v in (("train", train), ("val", val), ("test", test), ("ktest", ktest), ("unl", unl))}
    found = {k: ~np.isnan(v[:, 0, 0]) for k, v in L.items()}
    out["5_hand_found_rate"] = {k: float(v.mean()) for k, v in found.items()}
    tr = found["train"]
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.9, colsample_bytree=0.8,
                        random_state=SEED, n_jobs=6)
    xgb.fit(hand_features(L["train"][tr]), Y["train"][tr])

    def lm_proba(k):
        p = np.full((len(L[k]), 26), np.nan)
        if found[k].any():
            p[found[k]] = xgb.predict_proba(hand_features(L[k][found[k]]))
        return p

    PL = {k: lm_proba(k) for k in ("val", "test", "ktest")}
    out["5_landmarks_xgboost"] = {
        s: {"on_images_with_a_hand": scores(Y[s][found[s]], PL[s][found[s]].argmax(1)),
            # an image with no hand found counts as wrong
            "on_all_images": float((np.where(found[s], np.nan_to_num(PL[s]).argmax(1), -1) == Y[s]).mean())}
        for s in ("test", "ktest")}
    print("landmarks", out["5_landmarks_xgboost"], flush=True)

    # 6. late fusion, weight chosen on validation
    def fuse(pc, pl, w):
        f = pc.copy()
        has = ~np.isnan(pl[:, 0])
        f[has] = (1 - w) * pc[has] + w * pl[has]
        return f

    grid = np.round(np.arange(0, 1.01, 0.05), 2)
    val_acc = [float((fuse(P4["val"], PL["val"], w).argmax(1) == Y["val"]).mean()) for w in grid]
    w_best = float(grid[int(np.argmax(val_acc))])
    out["6_fusion"] = {"landmark_weight": w_best, "val_curve": dict(zip(map(float, grid), val_acc)),
                       **{s: scores(Y[s], fuse(P4[s], PL[s], w_best).argmax(1)) for s in ("test", "ktest")}}
    print("fusion", w_best, {s: out["6_fusion"][s] for s in ("test", "ktest")}, flush=True)

    confusion_plot(Y["test"], P4["test"].argmax(1), "EfficientNet-B0 (semi-supervised), held-out test", FIGS / "image_confusion_cnn.png")
    confusion_plot(Y["test"], fuse(P4["test"], PL["test"], w_best).argmax(1), "Fusion (CNN + landmarks), held-out test", FIGS / "image_confusion_fusion.png")
    plot_summary(out)

    # Exports for the app.
    MODELS.mkdir(exist_ok=True)
    cnn2.eval()
    torch.onnx.export(cnn2, torch.randn(1, 3, 224, 224), MODELS / "letters_cnn.onnx", input_names=["image"],
                      output_names=["logits"], dynamic_axes={"image": {0: "n"}}, opset_version=17, dynamo=False)
    xgb.save_model(MODELS / "letters_landmarks_xgb.json")
    np.save(MODELS / "letters_landmark_weight.npy", np.array([w_best]))
    out["seconds"] = round(time.time() - t0)
    (RESULTS / "image_results.json").write_text(json.dumps(out, indent=1))
    print("done in", out["seconds"], "s")


def plot_summary(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [("Pixels +\nLogReg", out["1_pixels_logreg"]), ("EffNet\nlinear probe", out["2_effnet_linear_probe"]),
            ("EffNet\nfine-tuned", out["3_effnet_finetuned"]), ("EffNet +\npseudo-labels", out["4_effnet_semi_supervised"]),
            ("Landmarks\nXGBoost", {s: {"accuracy": out["5_landmarks_xgboost"][s]["on_all_images"]} for s in ("test", "ktest")}),
            ("Fusion", out["6_fusion"])]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, 3.6))
    t = [r["test"]["accuracy"] * 100 for _, r in rows]
    k = [r["ktest"]["accuracy"] * 100 for _, r in rows]
    ax.bar(x - 0.2, t, 0.4, label="Held-out test (260)", color="#2f6fed")
    ax.bar(x + 0.2, k, 0.4, label="Kaggle test photos (26)", color="#e3a23b")
    for xi, a, b in zip(x, t, k):
        ax.text(xi - 0.2, a + 1, f"{a:.0f}", ha="center", fontsize=8)
        ax.text(xi + 0.2, b + 1, f"{b:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x, [n for n, _ in rows], fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "image_accuracy.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
