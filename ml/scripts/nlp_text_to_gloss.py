"""English word -> sign, learned from the 1910 manual (PDF / NLP).

Pipeline: PDF entries (pdf_entries.py) -> words -> BERT WordPiece tokens -> token
IDs -> 768-dim vectors. A word is mapped to the sign whose vector is nearest.

Training data (authentic): the manual groups words that share one sign
("Must, Need, Have to" = one sign) and describes each sign in English. The model
is fine-tuned so that a word lands near its own sign's other words and description.

Two evaluations
  A. Manual synonyms, 5-fold by entry. Query: a word from an entry that lists
     several words; answer: that entry. Candidates: all 900+ entries. The fold's
     test entries are never seen in training.
  B. The app's own vocabulary. Query: an English spelling the app maps to a sign
     (e.g. "afraid" -> SCARED); candidates: the app's 476 signs. These pairs were
     written for the app, not taken from a published source, and none is used for
     training, so this measures transfer to the app.

Methods compared: exact match after stemming (baseline), character n-gram TF-IDF,
frozen BERT, fine-tuned BERT.

Outputs: results/nlp_results.json, figures/nlp_*.png, models/bert-sign-encoder/,
data/processed/nlp_synonyms_for_app.json
"""
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
BERT = ROOT / "models" / "bert-base-uncased"
ENTRIES = ROOT / "data" / "processed" / "pdf" / "entries.csv"
APP_VOCAB = ROOT / "data" / "processed" / "app_vocab.json"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
SEED = 361
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(3)


# ── data ─────────────────────────────────────────────────────────────────────

def load_entries():
    df = pd.read_csv(ENTRIES, keep_default_na=False)
    df["words"] = df["headwords"].str.split("|")
    return df


def stem(w):
    w = w.lower().strip()
    for suf in ("ingly", "edly", "ness", "ment", "ship", "ing", "ies", "ied", "ed", "ly", "es", "er", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def app_benchmark():
    """(query word, sign) pairs from the app: English spellings that are not the sign's own name."""
    v = json.loads(APP_VOCAB.read_text())
    signs = [s.lower() for s in v["signs"]]
    urdu_like = re.compile(r"(aa|ee|oo|kh|gh|bh|dh|jh|th[aeiou]$|[^a-z])")
    pairs = []
    for word, sign in v["forms"].items():
        s = sign.lower()
        if word == s or stem(word) == stem(s):
            continue
        if urdu_like.search(word) and word not in ENGLISH_WITH_DIGRAPHS:
            continue  # romanised Urdu spellings are not an English-synonym test
        pairs.append((word, s))
    return signs, pairs


# English words that the Urdu filter above would wrongly drop.
ENGLISH_WITH_DIGRAPHS = {"bathroom", "breakfast", "good", "food", "book", "cook", "school", "room", "soon", "tooth",
                         "teeth", "feel", "need", "week", "sleep", "free", "green", "keep", "meet", "street", "three",
                         "thank", "think", "thing", "math", "both", "health", "father", "mother", "brother", "other",
                         "weather", "together", "nothing", "anything", "something", "everything", "teacher", "knee"}


# ── encoders ─────────────────────────────────────────────────────────────────

class Encoder:
    """BERT with mean pooling over WordPiece tokens -> L2-normalised 768-dim vectors."""

    def __init__(self, path=BERT):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path)

    def encode(self, texts, max_len=48, batch=64, grad=False):
        outs = []
        for i in range(0, len(texts), batch):
            enc = self.tok(texts[i:i + batch], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            with torch.set_grad_enabled(grad):
                h = self.model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1)
            outs.append(F.normalize(v, dim=-1))
        return torch.cat(outs)


def finetune(enc, train_df, epochs=4, lr=3e-5, batch=32, train_layers=4, log=None):
    """Contrastive fine-tuning: a word should be nearest its own sign (in-batch negatives).

    Positives: (word, the sign's other words + description) for every word of every training entry.
    Only the top `train_layers` transformer layers are trained (CPU budget, and less overfitting on ~1k entries).
    """
    model = enc.model
    for p in model.parameters():
        p.requires_grad = False
    for layer in model.encoder.layer[-train_layers:]:
        for p in layer.parameters():
            p.requires_grad = True
    pairs = []
    for idx, e in train_df.iterrows():
        for w in e["words"]:
            pairs.append((w, sign_text(e, exclude=w), idx))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    model.train()
    losses = []
    for ep in range(epochs):
        random.shuffle(pairs)
        for i in range(0, len(pairs) - 1, batch):
            chunk = pairs[i:i + batch]
            q = enc.encode([a for a, _, _ in chunk], max_len=12, batch=len(chunk), grad=True)
            k = enc.encode([b for _, b, _ in chunk], max_len=64, batch=len(chunk), grad=True)
            logits = q @ k.T / 0.05
            # Two words of the same entry in one batch are not negatives of each other.
            same = torch.tensor([[x[2] == y[2] for y in chunk] for x in chunk])
            logits = logits.masked_fill(same & ~torch.eye(len(chunk), dtype=torch.bool), -1e4)
            loss = F.cross_entropy(logits, torch.arange(len(chunk)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if log is not None:
            log.append({"epoch": ep + 1, "loss": float(np.mean(losses[-max(1, len(pairs) // batch):]))})
    model.eval()
    return losses


def sign_text(e, exclude=None):
    """How a sign is represented as a candidate: its other words, then its description."""
    others = [w for w in e["words"] if w != exclude]
    return (", ".join(others) + ". " if others else "") + e["description"][:240]


# ── scoring ──────────────────────────────────────────────────────────────────

def rank_metrics(ranks):
    r = np.array(ranks)
    return {"top1": float(np.mean(r == 1)), "top5": float(np.mean(r <= 5)), "mrr": float(np.mean(1.0 / r)), "n": int(len(r))}


def rank_of(scores, gold):
    order = np.argsort(-scores, kind="stable")
    return int(np.where(order == gold)[0][0]) + 1


def eval_manual_fold(df, test_ids, method, enc=None):
    """Synonym queries from the test entries (every word but the first), against every entry.

    A candidate entry is shown as its first word plus its description (the query
    word itself is never in it); the exact-match baseline sees the first word only.
    """
    heads = [e["words"][0] for _, e in df.iterrows()]
    cands = [sign_text_head(e) for _, e in df.iterrows()]
    queries = [(w, idx) for idx in test_ids for w in df.loc[idx, "words"][1:]]
    ranks = []
    if method == "exact":
        hs = [stem(h) for h in heads]
        for w, gold in queries:
            sc = np.array([1.0 if stem(w) == h else 0.0 for h in hs])
            ranks.append(rank_of(sc, gold) if sc[gold] > 0 else len(heads))
    elif method == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True).fit(cands)
        H = vec.transform(cands)
        for w, gold in queries:
            ranks.append(rank_of((vec.transform([w]) @ H.T).toarray()[0], gold))
    else:
        H = enc.encode(cands, max_len=64).numpy()
        Q = enc.encode([w for w, _ in queries], max_len=12).numpy()
        for (w, gold), q in zip(queries, Q):
            ranks.append(rank_of(H @ q, gold))
    return ranks


def sign_text_head(e):
    return e["words"][0] + ". " + e["description"][:240]


def app_sign_texts(signs, df):
    """Each app sign as text: its name, plus the manual's description when the manual has that word.

    Only the manual's text is added (never the app's own spellings, which are the test queries).
    """
    desc = {}
    for _, e in df.iterrows():
        for w in e["words"]:
            desc.setdefault(w, e["description"][:240])
    return [f"{s}. {desc[s]}" if s in desc else s for s in signs]


def eval_app(signs, pairs, method, enc=None, texts=None):
    idx = {s: i for i, s in enumerate(signs)}
    texts = texts or signs
    ranks = []
    if method == "exact":
        st = [stem(s) for s in signs]
        for w, s in pairs:
            sc = np.array([1.0 if stem(w) == x else 0.0 for x in st])
            ranks.append(rank_of(sc, idx[s]) if sc[idx[s]] > 0 else len(signs))
    elif method == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True).fit(texts)
        S = vec.transform(texts)
        for w, s in pairs:
            ranks.append(rank_of((vec.transform([w]) @ S.T).toarray()[0], idx[s]))
    else:
        S = enc.encode(texts, max_len=64).numpy()
        Q = enc.encode([w for w, _ in pairs], max_len=12).numpy()
        for (w, s), q in zip(pairs, Q):
            ranks.append(rank_of(S @ q, idx[s]))
    return ranks


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    RESULTS.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    df = load_entries()
    syn_ids = [i for i, e in df.iterrows() if len(e["words"]) > 1]
    random.Random(SEED).shuffle(syn_ids)
    folds = [syn_ids[k::5] for k in range(5)]
    out = {"entries": int(len(df)), "entries_with_synonyms": len(syn_ids), "manual": {}, "app": {}, "tokenization": {}}

    # Tokenization statistics (PDF -> words -> tokens -> IDs).
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BERT)
    words = sorted({w for ws in df["words"] for w in ws})
    pieces = [tok.tokenize(w) for w in words]
    desc_ids = [tok(d, add_special_tokens=False)["input_ids"] for d in df["description"]]
    out["tokenization"] = {
        "unique_headwords": len(words),
        "headwords_single_token": float(np.mean([len(p) == 1 for p in pieces])),
        "mean_wordpieces_per_headword": float(np.mean([len(p) for p in pieces])),
        "mean_tokens_per_description": float(np.mean([len(x) for x in desc_ids])),
        "example": {w: {"tokens": tok.tokenize(w), "ids": tok(w, add_special_tokens=False)["input_ids"]}
                    for w in ["thank", "weary", "physician", "cabinet-maker", "presbyterian"]},
    }
    print(json.dumps(out["tokenization"], indent=1)[:900])

    # A. manual synonyms, 5-fold
    per = {m: [] for m in ("exact", "tfidf", "bert_frozen", "bert_finetuned")}
    frozen = Encoder()
    curves = []
    for k, test in enumerate(folds):
        t0 = time.time()
        per["exact"] += eval_manual_fold(df, test, "exact")
        per["tfidf"] += eval_manual_fold(df, test, "tfidf")
        per["bert_frozen"] += eval_manual_fold(df, test, "bert", frozen)
        tuned = Encoder()
        log = []
        finetune(tuned, df.drop(index=test), log=log)
        curves.append(log)
        per["bert_finetuned"] += eval_manual_fold(df, test, "bert", tuned)
        print(f"fold {k + 1}/5 done in {time.time() - t0:.0f}s", flush=True)
    out["manual"] = {m: rank_metrics(r) for m, r in per.items()}
    out["manual_loss_curves"] = curves

    # B. the app's vocabulary, with a model fine-tuned on ALL manual entries
    signs, pairs = app_benchmark()
    texts = app_sign_texts(signs, df)
    out["app_signs_with_manual_description"] = int(sum(t != s for t, s in zip(texts, signs)))
    tuned = Encoder()
    final_log = []
    finetune(tuned, df, log=final_log)
    out["final_loss_curve"] = final_log
    app_ranks = {
        "exact": eval_app(signs, pairs, "exact"),
        "tfidf": eval_app(signs, pairs, "tfidf", texts=texts),
        "bert_frozen": eval_app(signs, pairs, "bert", frozen, texts=texts),
        "bert_finetuned": eval_app(signs, pairs, "bert", tuned, texts=texts),
    }
    out["app"] = {m: rank_metrics(r) for m, r in app_ranks.items()}
    out["app_pairs"] = len(pairs)

    # Confidence threshold for using a "similar" sign in the app: the lowest
    # similarity at which the fine-tuned model's top-1 is right >= 90% of the time.
    S = tuned.encode(texts, max_len=64).numpy()
    Q = tuned.encode([w for w, _ in pairs], max_len=12).numpy()
    sims = Q @ S.T
    best = sims.argmax(1)
    conf = sims.max(1)
    correct = np.array([signs[b] == s for b, (_, s) in zip(best, pairs)])
    thr = None
    for t in np.arange(0.30, 0.99, 0.01):
        sel = conf >= t
        if sel.sum() >= 20 and correct[sel].mean() >= 0.90:
            thr = float(round(t, 2))
            break
    out["app_threshold"] = {"similarity": thr,
                            "precision_at_threshold": float(correct[conf >= thr].mean()) if thr else None,
                            "coverage_at_threshold": float((conf >= thr).mean()) if thr else None}
    examples = []
    for (w, s), b, c in zip(pairs, best, conf):
        examples.append({"word": w, "gold": s, "predicted": signs[b], "similarity": round(float(c), 3)})
    out["app_examples"] = sorted(examples, key=lambda x: -x["similarity"])

    tuned.model.save_pretrained(ROOT / "models" / "bert-sign-encoder")
    tuned.tok.save_pretrained(ROOT / "models" / "bert-sign-encoder")
    export_for_app(tuned, signs, texts, thr, json.loads(APP_VOCAB.read_text())["forms"])
    (RESULTS / "nlp_results.json").write_text(json.dumps(out, indent=1))
    plot(out)
    print(json.dumps({"manual": out["manual"], "app": out["app"], "threshold": out["app_threshold"]}, indent=1))


def export_for_app(enc, signs, texts, thr, forms):
    """Precompute word -> similar sign for English words the app does not know.

    The browser cannot run a 440 MB BERT, so the model's decisions are computed
    here for a fixed list of words (BERT's own whole-word vocabulary, letters only,
    3+ characters) and shipped as a small table. Only matches at or above the
    90%-precision threshold are kept.
    """
    if thr is None:
        return
    vocab = [w for w in enc.tok.vocab if re.fullmatch(r"[a-z]{3,}", w) and w not in forms]
    S = enc.encode(texts, max_len=64)
    table = {}
    for i in range(0, len(vocab), 512):
        chunk = vocab[i:i + 512]
        sims = enc.encode(chunk, max_len=12) @ S.T
        val, arg = sims.max(1)
        for w, v, a in zip(chunk, val.tolist(), arg.tolist()):
            if v >= thr:
                table[w] = [signs[a].upper(), round(v, 3)]
    path = ROOT / "data" / "processed" / "nlp_synonyms_for_app.json"
    path.write_text(json.dumps({"threshold": thr, "model": "bert-base-uncased fine-tuned on Long (1910) sign groups",
                                "words": dict(sorted(table.items()))}, indent=0))
    print(f"{len(table)} similar-sign suggestions written to {path}")


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = {"exact": "Exact match\n(baseline)", "tfidf": "Char n-gram\nTF-IDF", "bert_frozen": "BERT\n(frozen)",
             "bert_finetuned": "BERT\n(fine-tuned)"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, title in ((axes[0], "manual", "A. Manual synonyms (5-fold)"), (axes[1], "app", "B. App vocabulary (transfer)")):
        ms = list(names)
        t1 = [out[key][m]["top1"] * 100 for m in ms]
        t5 = [out[key][m]["top5"] * 100 for m in ms]
        x = np.arange(len(ms))
        ax.bar(x - 0.2, t1, 0.4, label="Top-1", color="#2f6fed")
        ax.bar(x + 0.2, t5, 0.4, label="Top-5", color="#9db8f5")
        for xi, v in zip(x, t1):
            ax.text(xi - 0.2, v + 1, f"{v:.0f}", ha="center", fontsize=8)
        ax.set_xticks(x, [names[m] for m in ms], fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "nlp_accuracy.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    for k, c in enumerate(out["manual_loss_curves"]):
        ax.plot([p["epoch"] for p in c], [p["loss"] for p in c], marker="o", label=f"fold {k + 1}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Contrastive loss")
    ax.set_title("BERT fine-tuning loss", fontsize=10)
    ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "nlp_loss.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
