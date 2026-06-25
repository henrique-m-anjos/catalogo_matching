# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pandas",
#   "numpy",
#   "rank-bm25",
# ]
# ///

import numpy as np
import pandas as pd
from pathlib import Path
from rank_bm25 import BM25Okapi

TOP_N = 5

root = Path(__file__).parent

antigo = pd.read_csv(root / "data/catálogo_antigo.csv").fillna("")
novo   = pd.read_csv(root / "data/catalogo_novo.csv").fillna("")

antigo["descrição_antiga"] = antigo["descrição_antiga"].str.lower().str.strip()
novo["descrição_nova"]     = novo["descrição_nova"].str.lower().str.strip()

print(f"Antigo: {len(antigo)} rows | Novo: {len(novo)} rows")

# ── Build BM25 index on novo corpus ───────────────────────────────────────────
corpus = [desc.split() for desc in novo["descrição_nova"]]
bm25   = BM25Okapi(corpus)
print("BM25 index built.")

# ── Score every antigo row against the full novo corpus ───────────────────────
def next5_string(scores):
    # ranks 2–6: skip rank 1 (best match, already in its own columns)
    top_idx = np.argsort(scores)[::-1][1:TOP_N + 1]
    parts = []
    for idx in top_idx:
        if scores[idx] == 0:
            continue
        parts.append(f"{novo.iloc[idx]['código_novo']}: {novo.iloc[idx]['descrição_nova']}")
    return "; ".join(parts)

rows = []
for i, row in antigo.iterrows():
    query  = row["descrição_antiga"].split()
    scores = bm25.get_scores(query)

    best_idx = int(np.argmax(scores))
    rows.append({
        "código_antigo":    row["código_antigo"],
        "descrição_antiga": row["descrição_antiga"],
        "código_novo":      novo.iloc[best_idx]["código_novo"],
        "descrição_nova":   novo.iloc[best_idx]["descrição_nova"],
        "matches_proximos": next5_string(scores),
    })

    if (i + 1) % 500 == 0:
        print(f"  processed {i + 1}/{len(antigo)} rows...")

result = pd.DataFrame(rows)
out_path = root / "outputs" / "bm25_all_matches.csv"
result.to_csv(out_path, index=False)
print(f"\nDone. {len(result)} rows saved to {out_path}")
