#!/usr/bin/env python3
"""
Build curated PAD-annotated emotion lexicon from DUTIR ontology,
filtered by jieba word frequency for quality.

Pipeline:
  1. Load jieba dict → common word set (freq ≥ 10, ~170k words)
  2. Parse DUTIR CSV → map categories → assign PAD
  3. Filter: intensity≥5, polarity≠0, 1-2 chars, EXISTS in jieba dict
  4. Merge with hand-curated seed lexicon (take PAD from DUTIR when both exist)
  5. Cap per category, output curated_lexicon.json (~150KB)
"""

import csv
import json
import os
import sys
from collections import Counter

# ── category map ──
CATEGORY_MAP = {
    "PA": ("joy",      0.78, 0.60, 0.70),
    "PH": ("joy",      0.82, 0.55, 0.72),
    "PE": ("joy",      0.75, 0.62, 0.65),
    "PD": ("joy",      0.75, 0.55, 0.68),
    "LE": ("joy",      0.80, 0.68, 0.68),
    "PB": ("joy",      0.72, 0.50, 0.62),
    "PG": ("joy",      0.70, 0.58, 0.60),
    "PK": ("joy",      0.68, 0.52, 0.58),
    "NG": ("anger",    0.12, 0.82, 0.75),
    "NH": ("anger",    0.15, 0.70, 0.65),
    "NL": ("anger",    0.18, 0.65, 0.60),
    "NB": ("sadness",  0.12, 0.15, 0.15),
    "PF": ("sadness",  0.15, 0.20, 0.20),
    "NI": ("fear",     0.12, 0.75, 0.12),
    "NJ": ("fear",     0.15, 0.60, 0.15),
    "ND": ("disgust",  0.12, 0.55, 0.55),
    "NK": ("disgust",  0.15, 0.50, 0.50),
    "PC": ("surprise", 0.55, 0.72, 0.42),
    "NN": ("neutral",  0.35, 0.30, 0.35),
    "NE": ("neutral",  0.38, 0.28, 0.38),
    "NA": ("neutral",  0.35, 0.30, 0.35),
    "NC": ("neutral",  0.45, 0.35, 0.40),
    "OTH": ("neutral", 0.50, 0.40, 0.50),
}

MAX_PER_CAT = 500
JIEBA_DICT_URL = "https://raw.githubusercontent.com/fxsjy/jieba/master/extra_dict/dict.txt.big"
MIN_FREQ = 20


def load_jieba_common() -> set[str]:
    """Load common Chinese words from jieba dictionary (freq ≥ MIN_FREQ)."""
    import urllib.request
    import io

    common: set[str] = set()
    tmp = "/tmp/jieba_dict.txt"

    # Try cached first
    if os.path.exists(tmp):
        print(f"  Using cached {tmp}")
        with open(tmp, encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip().split()
                if len(parts) >= 2 and int(parts[1]) >= MIN_FREQ:
                    common.add(parts[0])
        return common

    # Download
    print(f"  Downloading jieba dict...")
    try:
        with urllib.request.urlopen(JIEBA_DICT_URL, timeout=30) as resp:
            data = resp.read().decode("utf-8")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(data)
        for line in data.splitlines():
            parts = line.rstrip().split()
            if len(parts) >= 2 and int(parts[1]) >= MIN_FREQ:
                common.add(parts[0])
    except Exception as e:
        print(f"  WARNING: could not download jieba dict: {e}")
        print(f"  Falling back to all DUTIR entries (quality will be lower)")
        return None  # signal: no filter

    return common


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/DLUT-Emotionontology/情感词汇/情感词汇.csv"
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)

    # ── load common words ──
    common = load_jieba_common()
    if common is not None:
        print(f"  Loaded {len(common)} common words (freq ≥ {MIN_FREQ})")

    # ── parse DUTIR ──
    rows = []
    with open(csv_path, encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        raw_header = next(reader)
        seen_h = set()
        indices = [(i, h.strip()) for i, h in enumerate(raw_header)
                   if h.strip() and h.strip() not in seen_h and not seen_h.add(h.strip())]
        for line in reader:
            vals = [v.strip() for v in line]
            d = {name: vals[i] if i < len(vals) else "" for i, name in indices}
            word = d.get("词语", "").strip()
            cat = d.get("情感分类", "").strip().upper()
            if not word or not cat:
                continue
            try:
                intensity = float(d.get("强度", "") or 0)
            except ValueError:
                intensity = 0
            try:
                polarity = int(float(d.get("极性", "") or 0))
            except ValueError:
                polarity = 0
            rows.append({"word": word, "cat": cat, "intensity": intensity, "polarity": polarity})

    print(f"Parsed: {len(rows)} words")

    # ── filter ──
    filtered = []
    for r in rows:
        if r["intensity"] < 5:
            continue
        if r["polarity"] == 0:
            continue
        if not (1 <= len(r["word"]) <= 2):
            continue
        if common is not None and r["word"] not in common:
            continue
        filtered.append(r)

    print(f"Filtered (int≥5, pol≠0, len≤2, common): {len(filtered)}")

    # ── map to PAD ──
    best: dict[str, dict] = {}
    for r in filtered:
        info = CATEGORY_MAP.get(r["cat"])
        if info is None:
            continue
        our_cat, bv, ba, bd = info
        ni = (r["intensity"] - 1) / 8
        v = round(max(0.0, min(1.0, bv + (ni - 0.5) * 0.25)), 2)
        a = round(max(0.0, min(1.0, ba + (ni - 0.5) * 0.22)), 2)
        d = round(max(0.0, min(1.0, bd + (ni - 0.5) * 0.18)), 2)
        entry = {"word": r["word"], "v": v, "a": a, "d": d, "category": our_cat, "intensity": round(ni, 2)}
        if r["word"] not in best or r["intensity"] > best[r["word"]].get("_raw", 0):
            entry["_raw"] = r["intensity"]
            best[r["word"]] = entry

    curated = list(best.values())
    for e in curated:
        e.pop("_raw", None)
    curated.sort(key=lambda e: e["intensity"], reverse=True)
    print(f"Dedup: {len(curated)}")

    # ── cap ──
    counts: dict[str, int] = {}
    capped = []
    for e in curated:
        cat = e["category"]
        if counts.get(cat, 0) < MAX_PER_CAT:
            capped.append(e)
            counts[cat] = counts.get(cat, 0) + 1
    curated = capped
    print(f"Capped (max {MAX_PER_CAT}/cat): {len(curated)}")

    # ── stats ──
    cat_stats = Counter(e["category"] for e in curated)
    print("\nCategory distribution:")
    for c, n in cat_stats.most_common():
        print(f"  {c}: {n}")

    # ── output ──
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "curated_lexicon.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({
            "_source": "DUTIR (徐琳宏等, 情报学报 2008) + jieba freq filter",
            "_filter": f"int≥5, pol≠0, 1-2chars, jieba freq≥{MIN_FREQ}, max{MAX_PER_CAT}/cat",
            "_size": f"{len(curated)} words",
            "words": curated,
        }, fh, ensure_ascii=False, indent=2)
    print(f"\n✅ {out_path}  ({os.path.getsize(out_path):,} bytes)")


if __name__ == "__main__":
    main()
