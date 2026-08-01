# Lexical Search Baseline Before Embeddings

A small SQLite FTS5 experiment that turns “semantic search should be better” into a falsifiable comparison. It builds a BM25 lexical baseline over 20 documents, evaluates eight hand-authored queries with qrels, injects index loss, fails a release gate, and verifies exact recovery.

## Requirements

- CPython 3.11 or newer.
- SQLite compiled with FTS5 (included in common Python distributions).
- No package installation, model download, network service, or paid infrastructure.

Check FTS5 and run the lab:

```bash
python3 -c 'import sqlite3; print(sqlite3.sqlite_version)'
python3 -m unittest -v test_lab.py
python3 search_lab.py
```

Expected baseline:

```text
source_count = 20
index_count = 20
Recall@3 = 1.0
nDCG@3 = 0.958498
gate_passed = true
```

These values are deterministic for the fixed corpus, queries, qrels, and SQLite ranking expression in this repository. They are a lab control, not a general search-quality claim.

## What the metrics mean

- **Recall@3** asks whether the first three results contain all documents judged relevant for a query.
- **nDCG@3** rewards placing documents with higher relevance grades earlier.
- **Index completeness** compares source rows with distinct indexed document rows.

The release gate requires Recall@3 of `1.0`, nDCG@3 of at least `0.95`, and equal source/index counts. A later dense, hybrid, or reranking system should be compared against the same queries and judgments instead of against intuition.

## Smallest working implementation

`documents` is the source of truth. `documents_fts` is an external-content FTS5 index, and `documents_vocab` exposes indexed document counts. `rebuild()` derives the index from source rows. `search()` returns ranked document IDs; `evaluate()` computes per-query and macro metrics from the fixed qrels.

The corpus deliberately contains exact identifiers such as `ERR_CONN_RESET`, operational phrases, and pairs of similarly relevant documents. This gives lexical retrieval a fair production-shaped workload instead of using only paraphrase-friendly questions.

## Failure injection and recovery

The lab removes `DOC-007` from the FTS index while leaving the source document intact:

```python
remove_from_index(db, "DOC-007")
```

The exact-identifier query then loses its only relevant result. Index count drops to 19, Recall@3 and nDCG@3 fall, and the gate fails. Recovery is a deterministic rebuild from the source table:

```python
rebuild(db)
```

The recovered evaluation must equal the complete baseline object, including rankings and per-query metrics. This is stronger than checking only that the missing row reappeared.

## Negative result: the lexical ceiling

```python
search(db, "automobile")
```

returns no results because the corpus does not contain that token. The miss is useful evidence: lexical retrieval handles exact terms and identifiers well, but it does not infer arbitrary synonymy. That creates a specific hypothesis for a later dense or hybrid experiment.

## Production gap

This lab keeps the corpus and qrels in code so every reader gets the same control. A production search system additionally needs representative query sampling, judgment guidelines and reviewer agreement, segmented metrics, authorization filtering, analyzers/tokenization decisions, ingestion reconciliation, freshness SLOs, schema migrations, query-time limits, click-bias controls, and rollback criteria.

Do not add embeddings until the candidate system can be tested against a stable lexical control. If dense retrieval improves paraphrases but regresses exact identifiers, hybrid fusion or query routing may be the better decision than replacement.

## Cleanup

The demo and test use temporary SQLite files and remove them automatically. If adapting the lab to a persistent path, stop writers and remove the database plus any `-wal` and `-shm` files.

## Related article

This repository is the repeatable evidence artifact for “Build a Lexical Search Baseline Before Adding Embeddings.”

