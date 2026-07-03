# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pandas",
#   "numpy",
#   "scikit-learn",
#   "rank-bm25",
#   "openpyxl",
# ]
# ///

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from rank_bm25 import BM25Okapi

TOP_N      = 6
CHUNK_SIZE = 500

root = Path(__file__).parent

antigo = pd.read_csv(root / "data/catálogo_antigo.csv").fillna("")
novo   = pd.read_csv(root / "data/catálogo_novo.csv").fillna("")

antigo["descrição_antiga"] = antigo["descrição_antiga"].str.lower().str.strip()
novo["descrição_nova"]     = novo["descrição_nova"].str.lower().str.strip()

print(f"Antigo: {len(antigo)} rows | Novo: {len(novo)} rows")

# ── Jaccard ───────────────────────────────────────────────────────────────────
print("Running Jaccard...")

all_texts  = list(antigo["descrição_antiga"]) + list(novo["descrição_nova"])
vectorizer = CountVectorizer(binary=True, analyzer="word")
vectorizer.fit(all_texts)

A = vectorizer.transform(antigo["descrição_antiga"])
B = vectorizer.transform(novo["descrição_nova"])

A_sum = np.array(A.sum(axis=1)).flatten()
B_sum = np.array(B.sum(axis=1)).flatten()

j_best_idx    = np.zeros(len(antigo), dtype=int)
j_topn_idx    = np.zeros((len(antigo), TOP_N), dtype=int)
j_topn_scores = np.zeros((len(antigo), TOP_N))

for start in range(0, len(antigo), CHUNK_SIZE):
    end     = min(start + CHUNK_SIZE, len(antigo))
    inter   = A[start:end].dot(B.T).toarray()
    union   = A_sum[start:end, None] + B_sum[None, :] - inter
    jaccard = np.where(union > 0, inter / union, 0.0)

    top_i = np.argsort(jaccard, axis=1)[:, -TOP_N:][:, ::-1]
    j_topn_idx[start:end]    = top_i
    j_topn_scores[start:end] = jaccard[np.arange(end - start)[:, None], top_i]
    j_best_idx[start:end]    = top_i[:, 0]

    if (start // CHUNK_SIZE) % 5 == 0:
        print(f"  processed {end}/{len(antigo)} rows...")

def jaccard_outros(row_idx):
    parts = []
    for rank in range(1, TOP_N):
        ni    = j_topn_idx[row_idx, rank]
        score = j_topn_scores[row_idx, rank]
        if score == 0:
            continue
        parts.append(f"{novo.iloc[ni]['código_novo']}: {novo.iloc[ni]['descrição_nova']}")
    return "; ".join(parts)

# ── BM25 ──────────────────────────────────────────────────────────────────────
print("Running BM25...")

corpus = [desc.split() for desc in novo["descrição_nova"]]
bm25   = BM25Okapi(corpus)

def bm25_outros(scores):
    top_idx = np.argsort(scores)[::-1][1:TOP_N]
    parts = []
    for idx in top_idx:
        if scores[idx] == 0:
            continue
        parts.append(f"{novo.iloc[idx]['código_novo']}: {novo.iloc[idx]['descrição_nova']}")
    return "; ".join(parts)

bm25_best_idx   = np.zeros(len(antigo), dtype=int)
bm25_outros_col = []

for i, row in antigo.iterrows():
    query  = row["descrição_antiga"].split()
    scores = bm25.get_scores(query)
    best   = int(np.argmax(scores))
    bm25_best_idx[i] = best
    bm25_outros_col.append(bm25_outros(scores))

    if (i + 1) % 500 == 0:
        print(f"  processed {i + 1}/{len(antigo)} rows...")

# ── Build output ──────────────────────────────────────────────────────────────
rows = []
for i in range(len(antigo)):
    ji = j_best_idx[i]
    bi = bm25_best_idx[i]
    rows.append({
        "código_antigo":      antigo.iloc[i]["código_antigo"],
        "descrição_antiga":   antigo.iloc[i]["descrição_antiga"],
        "código_jaccard":     novo.iloc[ji]["código_novo"],
        "descrição_jaccard":  novo.iloc[ji]["descrição_nova"],
        "código_bm25":        novo.iloc[bi]["código_novo"],
        "descrição_bm25":     novo.iloc[bi]["descrição_nova"],
        "outros_jaccard":     jaccard_outros(i),
        "outros_bm25":        bm25_outros_col[i],
    })

result   = pd.DataFrame(rows)
out_path = root / "outputs" / "matches.xlsx"
result.to_excel(out_path, index=False)
print(f"\nDone. {len(result)} rows saved to {out_path}")
