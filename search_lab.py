#!/usr/bin/env python3
"""SQLite FTS5 retrieval baseline reconstructed from the existing draft."""

import math
import sqlite3
import tempfile
from pathlib import Path

DOCS = [
    ("DOC-001", "Idempotent ticket mutation", "Safe retry for a ticket mutation uses one idempotency key and one durable result."),
    ("DOC-002", "Canonical retry intent", "Safe retry for a ticket mutation binds canonical intent to the request key."),
    ("DOC-003", "Lexical search baseline", "A lexical search BM25 baseline measures retrieval before embeddings."),
    ("DOC-004", "Lexical search judgments", "A lexical search BM25 baseline uses qrels and offline evaluation."),
    ("DOC-005", "RAG index drift", "Detect stale deleted and duplicate chunks in a RAG index."),
    ("DOC-006", "RAG ingestion reconciliation", "Repair stale deleted and duplicate chunks during RAG ingestion."),
    ("DOC-007", "Connection reset runbook", "ERR_CONN_RESET identifies a TCP connection reset while reading a response."),
    ("DOC-008", "Incomplete agent trace", "Reject an agent trace missing a tool result before evaluation."),
    ("DOC-009", "Trace evaluation gate", "An agent trace missing a tool result must not enter evaluation."),
    ("DOC-010", "Kubernetes agent sandbox", "Use Kubernetes isolation for an untrusted agent sandbox."),
    ("DOC-011", "Agent runtime isolation", "Kubernetes isolation constrains an untrusted agent sandbox runtime."),
    ("DOC-012", "Recover an expired lease", "Recover an expired agent lease by reclaiming work after its deadline."),
    ("DOC-013", "Lease heartbeat recovery", "Recover an expired agent lease with heartbeat expiry and a new owner."),
    ("DOC-014", "Agent SLO error budget", "Define an agent SLO and burn its error budget with failed runs."),
    ("DOC-015", "SLO release decision", "Use an agent SLO error budget to block an unsafe release."),
    ("DOC-016", "Hybrid retrieval", "Fuse sparse and dense rankings after measuring both controls."),
    ("DOC-017", "Reranking latency", "Measure ranking quality against latency before adding a reranker."),
    ("DOC-018", "Prompt approval", "Require approval before a consequential agent action."),
    ("DOC-019", "Canary rollback", "Canary an agent policy change and retain a rollback path."),
    ("DOC-020", "Document authorization", "Apply document authorization before retrieval results reach generation."),
]

QUERIES = {
    "Q1": '"ERR_CONN_RESET"',
    "Q2": "recover expired agent lease",
    "Q3": "safe retry ticket mutation",
    "Q4": "lexical search BM25 baseline",
    "Q5": "RAG stale deleted duplicate chunks",
    "Q6": "agent trace missing tool result evaluation",
    "Q7": "Kubernetes isolation untrusted agent sandbox",
    "Q8": "agent SLO error budget",
}

QRELS = {
    "Q1": {"DOC-007": 3},
    "Q2": {"DOC-012": 3, "DOC-013": 2},
    "Q3": {"DOC-001": 3, "DOC-002": 2},
    "Q4": {"DOC-003": 3, "DOC-004": 2},
    "Q5": {"DOC-005": 3, "DOC-006": 2},
    "Q6": {"DOC-008": 3, "DOC-009": 2},
    "Q7": {"DOC-010": 3, "DOC-011": 2},
    "Q8": {"DOC-014": 3, "DOC-015": 2},
}


def initialize(db):
    with sqlite3.connect(db) as conn:
        conn.executescript("""
        CREATE TABLE documents(doc_id TEXT UNIQUE NOT NULL, title TEXT NOT NULL, body TEXT NOT NULL);
        CREATE VIRTUAL TABLE documents_fts USING fts5(
          title, body, content='documents', content_rowid='rowid'
        );
        CREATE VIRTUAL TABLE documents_vocab USING fts5vocab(documents_fts, 'instance');
        """)
        conn.executemany(
            "INSERT INTO documents(doc_id,title,body) VALUES(?,?,?)", DOCS
        )


def rebuild(db):
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO documents_fts(documents_fts) VALUES('rebuild')")


def search(db, query, limit=3):
    with sqlite3.connect(db) as conn:
        rows = conn.execute("""
          SELECT d.doc_id, d.title, documents_fts.rank
          FROM documents_fts JOIN documents d ON d.rowid=documents_fts.rowid
          WHERE documents_fts MATCH ? ORDER BY documents_fts.rank LIMIT ?
        """, (query, limit)).fetchall()
    return [{"id": doc_id, "title": title, "rank": round(rank, 9)}
            for doc_id, title, rank in rows]


def recall(retrieved, judgments):
    relevant = set(judgments)
    return len(relevant & set(retrieved)) / len(relevant)


def ndcg(retrieved, judgments, k=3):
    gains = [judgments.get(doc, 0) for doc in retrieved[:k]]
    dcg = sum((2 ** gain - 1) / math.log2(position + 2)
              for position, gain in enumerate(gains))
    ideal = sorted(judgments.values(), reverse=True)[:k]
    idcg = sum((2 ** gain - 1) / math.log2(position + 2)
               for position, gain in enumerate(ideal))
    return dcg / idcg


def evaluate(db):
    details = {}
    for query_id, query in QUERIES.items():
        ids = [row["id"] for row in search(db, query)]
        details[query_id] = {
            "retrieved": ids,
            "recall": round(recall(ids, QRELS[query_id]), 6),
            "ndcg": round(ndcg(ids, QRELS[query_id]), 6),
        }
    macro_recall = round(sum(x["recall"] for x in details.values()) / len(details), 6)
    macro_ndcg = round(sum(x["ndcg"] for x in details.values()) / len(details), 6)
    with sqlite3.connect(db) as conn:
        source_count = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
        index_count = conn.execute(
            "SELECT count(DISTINCT doc) FROM documents_vocab"
        ).fetchone()[0]
    return {
        "source_count": source_count,
        "index_count": index_count,
        "recall_at_3": macro_recall,
        "ndcg_at_3": macro_ndcg,
        "gate_passed": (
            macro_recall >= 1.0 and macro_ndcg >= 0.95
            and source_count == index_count
        ),
        "details": details,
    }


def remove_from_index(db, doc_id):
    with sqlite3.connect(db) as conn:
        conn.execute("""
          INSERT INTO documents_fts(documents_fts, rowid, title, body)
          SELECT 'delete', rowid, title, body FROM documents WHERE doc_id=?
        """, (doc_id,))


def demo():
    handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    handle.close()
    db = Path(handle.name)
    try:
        initialize(db)
        rebuild(db)
        baseline = evaluate(db)
        remove_from_index(db, "DOC-007")
        broken = evaluate(db)
        assert not broken["gate_passed"]
        rebuild(db)
        recovered = evaluate(db)
        assert recovered == baseline
        assert search(db, "automobile") == []
        print({"baseline": baseline, "broken": broken, "recovered": recovered})
    finally:
        db.unlink(missing_ok=True)


if __name__ == "__main__":
    demo()

