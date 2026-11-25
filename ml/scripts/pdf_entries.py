"""PDF -> clean text -> dictionary entries.

Source: J. Schuyler Long, "The Sign Language: A Manual of Signs" (1910), the
scanned copy on archive.org. Its text layer is the scan's OCR, so it is noisy:
running page headers, words split across lines ("fore-\\nfinger"), the letter O
read as the digit 0, stray symbols, and plate pages of picture captions.

Each sign in the manual is one paragraph:
    Must, Need, Have to.—Crook the forefinger of the right "G" hand, ...
The words before the dash all share the one sign; the text after it says how
to make it. Those groups of words are what the NLP model learns from.

Output
  data/processed/pdf/pages_raw.json       text of every page, as extracted
  data/processed/pdf/clean_text.txt       after cleaning
  data/processed/pdf/entries.csv          one row per sign
  data/processed/pdf/cleaning_report.json what each cleaning step changed
"""
import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data" / "raw" / "long_1910" / "sign_language_manual_long_1910.pdf"
OUT = ROOT / "data" / "processed" / "pdf"

HEADER = re.compile(r"^\s*(\d+\s+)?(A\s+Manual\s+of\s+Signs|The\s+Sign\s+Language)\.?(\s+\d+)?\s*$", re.I)
SOFT_HYPHEN = "￾"  # how the OCR marks a word broken at the end of a line
# A paragraph that starts a sign: a few capitalised words, then a dash.
ENTRY = re.compile(r"^(?P<head>[\"“A-Z][^—]{0,90}?)\s*[.,:;]?\s*[—–]\s*(?P<body>.*)$")
# The book ends with an index: "Brother (m, 37) 27", "Bear, carry 55" (words, plate, pages).
INDEX_FIRST_PAGE = 199
INDEX_LINE = re.compile(r"^(?P<words>[A-Za-z][A-Za-z' ,-]{0,40}?)[\s.]+(?P<pages>\d{1,3}(\s*,\s*\d{1,3})*)$")
DASH_FIX = re.compile(r"\s*-\s*-\s*|(?<=[a-z.])--(?=[A-Z])")
HANDSHAPE = re.compile(r"[\"“]\s*([A-Z0-9])\s*[\"”]\s*hand", re.I)
BODY_PLACES = ["forehead", "eyes", "eye", "nose", "mouth", "lips", "chin", "cheek", "ear", "head", "face",
               "neck", "shoulder", "breast", "chest", "heart", "stomach", "waist", "side", "palm", "wrist", "elbow", "arm"]


def extract_pages():
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(str(PDF))
    return [doc[i].get_textpage().get_text_range() for i in range(len(doc))]


def alpha_ratio(s):
    s = s.strip()
    return sum(c.isalpha() for c in s) / max(1, len(s))


def clean(pages, report):
    """Page texts -> one list of clean lines, tagged with their page number."""
    lines = []
    for pno, text in enumerate(pages, start=1):
        if "Plate" in text[:40] and alpha_ratio(text) < 0.75:
            report["plate_pages_dropped"] += 1
            continue
        for raw in text.replace("\r", "").split("\n"):
            line = raw.strip()
            if not line:
                continue
            if HEADER.match(line):
                report["headers_removed"] += 1
                continue
            if len(line) <= 3 or alpha_ratio(line) < 0.5:
                report["noise_lines_removed"] += 1
                continue
            lines.append([pno, line])

    # Join words broken across lines: "fore￾" + "finger" -> "forefinger".
    joined = []
    for pno, line in lines:
        if joined and (joined[-1][1].endswith(SOFT_HYPHEN) or re.search(r"[a-z]-$", joined[-1][1])):
            joined[-1][1] = joined[-1][1].rstrip(SOFT_HYPHEN).rstrip("-") + line
            report["hyphenations_joined"] += 1
        else:
            joined.append([pno, line])

    out = []
    for pno, line in joined:
        before = line
        line = line.replace(SOFT_HYPHEN, "")
        line = DASH_FIX.sub("—", line)
        # The OCR reads the hand letter O as a zero: "0" hand -> "O" hand.
        line = re.sub(r"([\"“]\s*)0(\s*[\"”])", r"\1O\2", line)
        line = re.sub(r"[|^~*\\]", "", line)
        line = re.sub(r"\s{2,}", " ", line).strip()
        if line != before:
            report["lines_repaired"] += 1
        out.append((pno, line))
    return out


def split_head(head):
    """'May, Maybe, Perhaps, indicating probability' -> (['may','maybe','perhaps'], 'indicating probability')."""
    head = re.sub(r"\(.*?\)", "", head)
    words, notes = [], []
    for part in head.split(","):
        part = part.strip(" .\"“”'")
        if not part or part.lower() in ("etc", "or"):
            continue
        part = re.sub(r"^(to|the|a|an)\s+", "", part, flags=re.I)
        if part[:1].isupper() and len(part.split()) <= 4:
            words.append(part.lower())
        else:
            notes.append(part)
    return words, ", ".join(notes)


def parse_entries(lines):
    entries = []
    cur = None
    for pno, line in lines:
        m = ENTRY.match(line)
        if m:
            words, note = split_head(m.group("head"))
            if words:
                cur = {"page": pno, "headwords": words, "note": note, "description": m.group("body")}
                entries.append(cur)
                continue
        if cur is not None:
            cur["description"] += " " + line
    # The manual's front matter (introduction) has no dash paragraphs that pass,
    # but an entry whose description is far too short is an OCR fragment.
    entries = [e for e in entries if len(e["description"]) >= 25]
    for i, e in enumerate(entries):
        e["entry_id"] = i
        d = e["description"]
        e["handshapes"] = sorted({h.upper().replace("0", "O") for h in HANDSHAPE.findall(d)})
        e["places"] = sorted({p for p in BODY_PLACES if re.search(rf"\b{p}\b", d, re.I)})
        e["two_handed"] = bool(re.search(r"\bboth\b|\bleft\b.*\bright\b|\bthe hands\b", d, re.I))
    return entries


def parse_index(lines, first_page):
    """Words listed in the book's index, from `first_page` on."""
    words = set()
    for pno, line in lines:
        if pno < first_page:
            continue
        m = INDEX_LINE.match(re.sub(r"\(.*?\)", " ", line).strip())
        if m:
            for w in m.group("words").split(","):
                w = w.strip(" .").lower()
                if w:
                    words.add(w)
    return words


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pages = extract_pages()
    (OUT / "pages_raw.json").write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    report = Counter()
    lines = clean(pages, report)
    (OUT / "clean_text.txt").write_text("\n".join(f"{p}\t{l}" for p, l in lines), encoding="utf-8")
    entries = [e for e in parse_entries(lines) if e["page"] < INDEX_FIRST_PAGE]
    for i, e in enumerate(entries):
        e["entry_id"] = i
    index_words = parse_index(lines, INDEX_FIRST_PAGE)
    (OUT / "index_words.txt").write_text("\n".join(sorted(index_words)), encoding="utf-8")
    found = {w for e in entries for w in e["headwords"]}
    with open(OUT / "entries.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["entry_id", "page", "headwords", "note", "handshapes", "places", "two_handed", "description"])
        for e in entries:
            w.writerow([e["entry_id"], e["page"], "|".join(e["headwords"]), e["note"], "|".join(e["handshapes"]),
                        "|".join(e["places"]), int(e["two_handed"]), e["description"]])
    stats = dict(report)
    stats.update({
        "pages": len(pages),
        "raw_characters": sum(map(len, pages)),
        "clean_lines": len(lines),
        "entries": len(entries),
        "headwords": sum(len(e["headwords"]) for e in entries),
        "entries_with_synonyms": sum(len(e["headwords"]) > 1 for e in entries),
        "entries_with_handshape": sum(bool(e["handshapes"]) for e in entries),
        "index_words": len(index_words),
        "index_words_found_as_headwords": len(index_words & found),
    })
    (OUT / "cleaning_report.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
