# Catalog Matching

Faz a correspondência de cada linha do catálogo antigo com o catálogo novo e devolve o melhor match e os 5 mais próximos seguintes por linha. Estão disponíveis dois algoritmos: **Jaccard** e **BM25**.

## Configuração

Coloca os ficheiros de input dentro da pasta `data/` com os seguintes nomes:
```
catalogo_matching/
└── data/
    ├── catálogo_antigo.csv   ← deve ter as colunas: código_antigo, descrição_antiga
    └── catalogo_novo.csv     ← deve ter as colunas: código_novo, descrição_nova
```

---

## Execução

### Instalações

```bash
pip install pandas scikit-learn numpy rank-bm25
```

### Jaccard

```bash
python3 calculate_jaccard_scores.py
```

Output: `outputs/jaccard_all_matches.csv`

---

### BM25

```bash
python3 calculate_bm25_scores.py
```

Output: `outputs/bm25_all_matches.csv`

---

## Colunas do output

| Coluna | Descrição |
|---|---|
| `código_antigo` | Código do catálogo antigo |
| `descrição_antiga` | Descrição do catálogo antigo |
| `código_novo` | Código do melhor match no catálogo novo |
| `descrição_nova` | Descrição do melhor match no catálogo novo |
| `matches_proximos` | Top 5 matches seguintes (ranks 2–6) no formato `código: descrição; ...` |
