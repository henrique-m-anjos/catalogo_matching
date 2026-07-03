# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pandas",
#   "scikit-learn",
#   "numpy",
# ]
# ///

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer

TOP_N      = 6   # best match + 5 proximos
CHUNK_SIZE = 500

root = Path(__file__).parent

antigo = pd.read_csv(root / "data/catálogo_antigo.csv").fillna("")
novo   = pd.read_csv(root / "data/catálogo_novo.csv").fillna("")

antigo["descrição_antiga"] = antigo["descrição_antiga"].str.lower().str.strip()
novo["descrição_nova"]     = novo["descrição_nova"].str.lower().str.strip()

print(f"Antigo: {len(antigo)} rows | Novo: {len(novo)} rows")

# ── Vectorize ─────────────────────────────────────────────────────────────────
all_texts = list(antigo["descrição_antiga"]) + list(novo["descrição_nova"])
vectorizer = CountVectorizer(binary=True, analyzer="word")
vectorizer.fit(all_texts)

A = vectorizer.transform(antigo["descrição_antiga"])
B = vectorizer.transform(novo["descrição_nova"])

A_sum = np.array(A.sum(axis=1)).flatten()
B_sum = np.array(B.sum(axis=1)).flatten()

# ── Compute best match + top N per antigo row ─────────────────────────────────
best_idx    = np.zeros(len(antigo), dtype=int)
best_score  = np.zeros(len(antigo))
topn_idx    = np.zeros((len(antigo), TOP_N), dtype=int)
topn_scores = np.zeros((len(antigo), TOP_N))

for start in range(0, len(antigo), CHUNK_SIZE):
    end     = min(start + CHUNK_SIZE, len(antigo))
    inter   = A[start:end].dot(B.T).toarray()
    union   = A_sum[start:end, None] + B_sum[None, :] - inter
    jaccard = np.where(union > 0, inter / union, 0.0)

    top_i = np.argsort(jaccard, axis=1)[:, -TOP_N:][:, ::-1]
    topn_idx[start:end]    = top_i
    topn_scores[start:end] = jaccard[np.arange(end - start)[:, None], top_i]

    best_idx[start:end]   = top_i[:, 0]
    best_score[start:end] = topn_scores[start:end, 0]

    if (start // CHUNK_SIZE) % 5 == 0:
        print(f"  processed {end}/{len(antigo)} rows...")

# ── Build output ──────────────────────────────────────────────────────────────
def next5_string(row_idx):
    # ranks 2–6: skip rank 0 (best match, already in its own columns)
    parts = []
    for rank in range(1, TOP_N):
        ni    = topn_idx[row_idx, rank]
        score = topn_scores[row_idx, rank]
        if score == 0:
            continue
        parts.append(f"{novo.iloc[ni]['código_novo']}: {novo.iloc[ni]['descrição_nova']}")
    return "; ".join(parts)

rows = []
for i in range(len(antigo)):
    ni = best_idx[i]
    rows.append({
        "código_antigo":    antigo.iloc[i]["código_antigo"],
        "descrição_antiga": antigo.iloc[i]["descrição_antiga"],
        "código_novo":      novo.iloc[ni]["código_novo"],
        "descrição_nova":   novo.iloc[ni]["descrição_nova"],
        "matches_proximos": next5_string(i),
    })

result = pd.DataFrame(rows)
out_path = root / "outputs" / "jaccard_all_matches.xlsx"
result.to_excel(out_path, index=False)
print(f"\nDone. {len(result)} rows saved to {out_path}")
