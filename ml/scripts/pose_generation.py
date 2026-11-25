"""Text -> sign motion for the avatar.

A gloss's text vector (the BERT sign encoder from nlp_text_to_gloss.py) is
decoded into the motion of a sign: 32 time steps of the dominant wrist and
index fingertip, the other wrist, and the dominant handshape (15 fingertip
distances), in the body coordinates of clean_landmarks.py. The decoder is
trained on real signers' landmarks with a masked mean squared error (a value
MediaPipe did not see is left out of the loss).

The decoder is scored on WLASL test videos (people and recordings it has not
seen, of signs it has seen) against:
  global mean       one average motion for every sign
  avatar            the app's hand-authored sign (export_avatar_trajectories.mjs)
  retrieval         the training recording of the sign closest to the others (a real motion)
  per-sign mean     the average of the sign's training recordings

Outputs: results/pose_results.json, figures/pose_*.png,
data/processed/learned_motion_for_app.json
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
ENCODER = ROOT / "models" / "bert-sign-encoder"
SEED = 361
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(4)

T = 32
PROD = ["dom_x", "dom_y", "dom_tip_x", "dom_tip_y", "oth_x", "oth_y"] + [f"dom_shape{k}" for k in range(63, 78)]
N_POS = 6


def load_sequences():
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from tabular_models import active_part, resample
    names = json.loads((PROC / "feature_names.json").read_text())
    videos = pd.read_csv(PROC / "videos.csv", dtype={"video_id": str})
    seqs = np.load(PROC / "sequences.npz")
    cols = [names.index(c) for c in PROD]
    Y = np.stack([resample(active_part(seqs[v], names), T)[:, cols] for v in videos["video_id"]])
    return videos, Y


def text_vectors(glosses):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ENCODER)
    model = AutoModel.from_pretrained(ENCODER).eval()
    with torch.no_grad():
        enc = tok(list(glosses), padding=True, return_tensors="pt")
        h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        v = (h * m).sum(1) / m.sum(1)
    return dict(zip(glosses, nn.functional.normalize(v, dim=-1).numpy()))


class Decoder(nn.Module):
    """Text vector + time step -> one frame of motion."""

    def __init__(self, d_text=768, d=len(PROD), h=256):
        super().__init__()
        self.text = nn.Sequential(nn.Linear(d_text, h), nn.GELU(), nn.Dropout(0.1))
        self.time = nn.Linear(16, h)
        self.body = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, d))

    def forward(self, txt):
        t = torch.linspace(0, 1, T)
        freq = torch.arange(8)
        pe = torch.cat([torch.sin(np.pi * (2 ** freq)[None] * t[:, None] / 2), torch.cos(np.pi * (2 ** freq)[None] * t[:, None] / 2)], 1)
        z = self.text(txt)[:, None, :] + self.time(pe)[None]
        return self.body(z)


def masked_mse(pred, target, cols=None):
    if cols is not None:
        pred, target = pred[..., cols], target[..., cols]
    m = ~np.isnan(target)
    return float(((np.nan_to_num(pred) - np.nan_to_num(target)) ** 2)[m].mean())


def main():
    RESULTS.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    t0 = time.time()
    videos, Y = load_sequences()
    tr = (videos["split"] == "train").to_numpy()
    va = (videos["split"] == "val").to_numpy()
    te = (videos["split"] == "test").to_numpy()
    glosses = sorted(videos["gloss"].unique())
    vec = text_vectors(glosses)
    Xtxt = np.stack([vec[g] for g in videos["gloss"]]).astype(np.float32)

    # Standardise targets with training statistics; NaN = not seen by MediaPipe.
    mu = np.nanmean(Y[tr].reshape(-1, len(PROD)), 0)
    sd = np.nanstd(Y[tr].reshape(-1, len(PROD)), 0) + 1e-3
    Z = (Y - mu) / sd

    model = Decoder()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    Xt, Zt = torch.tensor(Xtxt[tr]), torch.tensor(Z[tr], dtype=torch.float32)
    Mt = ~torch.isnan(Zt)
    Zt = torch.nan_to_num(Zt)
    Xv = torch.tensor(Xtxt[va])
    log, best, best_state = [], 1e9, None
    for ep in range(400):
        model.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(Xt), 64):
            idx = perm[i:i + 64]
            pred = model(Xt[idx] + 0.02 * torch.randn_like(Xt[idx]))
            loss = (((pred - Zt[idx]) ** 2) * Mt[idx]).sum() / Mt[idx].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            pv = model(Xv).numpy() * sd + mu
        v = masked_mse(pv, Y[va])
        log.append({"epoch": ep + 1, "train_mse_std_units": tot / len(Xt), "val_mse": v})
        if v < best:
            best, best_state = v, {k: x.clone() for k, x in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        gen = {g: model(torch.tensor(vec[g][None].astype(np.float32))).numpy()[0] * sd + mu for g in glosses}

    # Baselines from the training recordings (train + val).
    pool = tr | va
    global_mean = np.nanmean(Y[pool], 0)
    per_mean, medoid = {}, {}
    for g in glosses:
        idx = np.where(pool & (videos["gloss"] == g).to_numpy())[0]
        if len(idx) == 0:
            continue
        per_mean[g] = np.nanmean(Y[idx], 0)
        if len(idx) == 1:
            medoid[g] = Y[idx[0]]
            continue
        d = np.array([[masked_mse(Y[i], Y[j]) for j in idx] for i in idx])
        medoid[g] = Y[idx[int(np.nanargmin(np.nanmean(np.where(np.eye(len(idx)) > 0, np.nan, d), 1)))]]

    avatar = {}
    path = PROC / "avatar_trajectories.json"
    if path.exists():
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from tabular_models import resample
        for g, a in json.loads(path.read_text()).items():
            fr = np.array(a["frames"], dtype=np.float32)
            # Same "signing part" rule as for people: the dominant hand raised above the navel.
            up = np.where(fr[:, 1] > -1.0)[0]
            if len(up) >= 2:
                fr = fr[up[0]:up[-1] + 1]
            avatar[g] = resample(fr, T)

    test_idx = np.where(te)[0]
    methods = {"global_mean": lambda g: global_mean, "per_sign_mean": lambda g: per_mean.get(g),
               "retrieval": lambda g: medoid.get(g), "generated": lambda g: gen[g], "avatar": lambda g: avatar.get(g)}
    pos_cols, shape_cols = list(range(N_POS)), list(range(N_POS, len(PROD)))
    out = {"test_videos": int(len(test_idx)), "glosses": len(glosses), "avatar_glosses": len(avatar), "all": {}, "avatar_subset": {}}
    for subset in ("all", "avatar_subset"):
        ids = [i for i in test_idx if subset == "all" or videos["gloss"].iloc[i] in avatar]
        out[subset]["videos"] = len(ids)
        for name, f in methods.items():
            if subset == "all" and name == "avatar":
                continue
            pairs = [(f(videos["gloss"].iloc[i]), Y[i]) for i in ids]
            pairs = [(p, y) for p, y in pairs if p is not None]
            out[subset][name] = {
                "position_mse": float(np.mean([masked_mse(p, y, pos_cols) for p, y in pairs])),
                "handshape_mse": float(np.nanmean([masked_mse(p, y, shape_cols) for p, y in pairs])),
                "videos": len(pairs)}
    # Per-sign gap between the avatar and real signers (which hand-authored signs are furthest off).
    gap = {}
    for g in avatar:
        ids = [i for i in test_idx if videos["gloss"].iloc[i] == g]
        if ids:
            gap[g] = float(np.mean([masked_mse(avatar[g], Y[i], pos_cols) for i in ids]))
    out["avatar_position_mse_by_sign"] = dict(sorted(gap.items(), key=lambda kv: -kv[1]))
    out["curve"] = log[::10]
    out["seconds"] = round(time.time() - t0)
    (RESULTS / "pose_results.json").write_text(json.dumps(out, indent=1))
    plot(out, Y, videos, test_idx, gen, medoid, avatar)

    # For the app: the generated motion and the retrieved real motion of every sign the app also has.
    app = {g: {"sign": None, "generated": np.round(gen[g], 4).tolist(), "retrieved": np.round(np.nan_to_num(medoid[g], nan=-9), 4).tolist()}
           for g in glosses if g in medoid}
    if path.exists():
        for g, a in json.loads(path.read_text()).items():
            if g in app:
                app[g]["sign"] = a["sign"]
    (PROC / "learned_motion_for_app.json").write_text(json.dumps({"steps": T, "columns": PROD, "signs": app}))
    print(json.dumps({k: out[k] for k in ("all", "avatar_subset")}, indent=1))


def plot(out, Y, videos, test_idx, gen, medoid, avatar):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sub = out["avatar_subset"]
    names = [("global_mean", "Global mean"), ("avatar", "Avatar\n(hand-authored)"), ("retrieval", "Retrieval\n(real recording)"),
             ("per_sign_mean", "Per-sign mean"), ("generated", "Generated\n(BERT -> MLP, MSE)")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key, title in ((axes[0], "position_mse", "Hand positions"), (axes[1], "handshape_mse", "Handshape")):
        v = [sub[n][key] for n, _ in names]
        ax.bar(range(len(v)), v, color=["#9aa5b1", "#e3a23b", "#2f6fed", "#7fa7f2", "#1f9d5a"])
        for i, x in enumerate(v):
            ax.text(i, x, f"{x:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(v)), [l for _, l in names], fontsize=7)
        ax.set_ylabel("MSE vs unseen real signers (lower is better)", fontsize=8)
        ax.set_title(f"{title} ({sub['videos']} test videos of {out['avatar_glosses']} signs)", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "pose_mse.png", dpi=160)
    plt.close(fig)

    show = [g for g in ("drink", "thank you", "yes", "eat", "help", "go") if g in avatar][:3]
    if show:
        fig, axes = plt.subplots(1, len(show), figsize=(4 * len(show), 3.2), squeeze=False)
        t = np.linspace(0, 1, T)
        for ax, g in zip(axes[0], show):
            for i in [i for i in test_idx if videos["gloss"].iloc[i] == g][:4]:
                ax.plot(t, Y[i][:, 1], color="#9aa5b1", lw=1)
            ax.plot(t, avatar[g][:, 1], color="#e3a23b", lw=2, label="avatar")
            ax.plot(t, medoid[g][:, 1], color="#2f6fed", lw=2, label="retrieved")
            ax.plot(t, gen[g][:, 1], color="#1f9d5a", lw=2, label="generated")
            ax.set_title(f'"{g}": dominant wrist height', fontsize=9)
            ax.set_xlabel("time (normalised)")
            ax.set_ylabel("shoulder widths above shoulders", fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0][0].legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(FIGS / "pose_examples.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
