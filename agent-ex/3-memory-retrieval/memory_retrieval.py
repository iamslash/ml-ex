"""
Generative Agents style 3-axis memory retrieval:
  recency + importance + relevance (cosine similarity on bag-of-words embeddings).

Paper: Park et al., "Generative Agents: Interactive Simulacra of Human Behavior" (2023)
https://arxiv.org/abs/2304.03442
"""

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

# stdlib-only: no numpy required

# ---------------------------------------------------------------------------
# Vocabulary & embedding
# ---------------------------------------------------------------------------

# Fixed vocabulary built from all demo sentences (no external model needed).
_VOCAB: list[str] = []
_WORD2IDX: dict[str, int] = {}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def build_vocab(texts: list[str]) -> None:
    """Populate global vocabulary from a list of texts."""
    global _VOCAB, _WORD2IDX
    words: set[str] = set()
    for t in texts:
        words.update(_tokenize(t))
    _VOCAB = sorted(words)
    _WORD2IDX = {w: i for i, w in enumerate(_VOCAB)}


def embed(text: str) -> list[float]:
    """Bag-of-words embedding; returns a float32 vector of vocab length."""
    vec = [0.0] * len(_VOCAB)
    for word in _tokenize(text):
        idx = _WORD2IDX.get(word)
        if idx is not None:
            vec[idx] += 1.0
    return vec


# ---------------------------------------------------------------------------
# MemoryEntry
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    text: str
    timestamp: datetime
    importance_score: float          # raw 1-10
    embedding: list[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.embedding and _VOCAB:
            self.embedding = embed(self.text)


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

DECAY_FACTOR = 0.995   # per-hour exponential decay (matches Park et al.)

def recency_score(entry: MemoryEntry, now: datetime) -> float:
    """Exponential decay based on hours since memory was formed."""
    delta_hours = max((now - entry.timestamp).total_seconds() / 3600.0, 0.0)
    return float(DECAY_FACTOR ** delta_hours)


def importance_score(entry: MemoryEntry) -> float:
    """Normalize raw 1-10 importance to [0, 1]."""
    return (entry.importance_score - 1.0) / 9.0


def relevance_score(query_embedding: list[float], entry_embedding: list[float]) -> float:
    """Cosine similarity in [0, 1]; returns 0 for zero vectors."""
    dot = sum(a * b for a, b in zip(query_embedding, entry_embedding))
    qn = math.sqrt(sum(a * a for a in query_embedding))
    en = math.sqrt(sum(b * b for b in entry_embedding))
    if qn < 1e-9 or en < 1e-9:
        return 0.0
    return dot / (qn * en)


# ---------------------------------------------------------------------------
# MemoryRetriever
# ---------------------------------------------------------------------------

class MemoryRetriever:
    """Add memories and retrieve the top-k by hybrid score."""

    def __init__(self):
        self._memories: list[MemoryEntry] = []

    def add_memory(self, entry: MemoryEntry) -> None:
        # Re-embed with current vocab if embedding is stale / empty
        if len(entry.embedding) != len(_VOCAB):
            entry.embedding = embed(entry.text)
        self._memories.append(entry)

    def retrieve_top_k(
        self,
        query: str,
        k: int = 5,
        alpha: float = 1.0,   # weight for recency
        beta: float = 1.0,    # weight for importance
        gamma: float = 1.0,   # weight for relevance
        now: Optional[datetime] = None,
        verbose: bool = False,
    ) -> list[tuple[MemoryEntry, dict]]:
        """
        Returns list of (entry, score_dict) sorted by combined score descending.
        score_dict keys: recency, importance, relevance, combined
        """
        if now is None:
            now = datetime.now(timezone.utc)

        q_emb = embed(query)
        scored: list[tuple[MemoryEntry, dict]] = []

        for entry in self._memories:
            rec = recency_score(entry, now)
            imp = importance_score(entry)
            rel = relevance_score(q_emb, entry.embedding)
            combined = alpha * rec + beta * imp + gamma * rel
            scored.append((entry, {
                "recency": rec,
                "importance": imp,
                "relevance": rel,
                "combined": combined,
            }))

        scored.sort(key=lambda x: x[1]["combined"], reverse=True)
        return scored[:k]


# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------

def _dt(days_ago: float, hours_ago: float = 0.0) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days_ago, hours=hours_ago)


_DEMO_MEMORIES = [
    # (text, days_ago, importance 1-10)
    ("database connection timeout error occurred in production", 0.1,  9),
    ("fixed bug in user authentication login flow",             0.5,  7),
    ("attended weekly team standup meeting",                    1.0,  2),
    ("database schema migration completed successfully",        2.0,  6),
    ("wrote unit tests for the payment module",                 3.0,  5),
    ("coffee machine in the office is broken",                  3.5,  1),
    ("connection pool exhausted under heavy load",              5.0,  8),
    ("updated project README with new instructions",            7.0,  3),
    ("reviewed pull request for feature branch",               10.0,  4),
    ("database query performance degraded after index drop",   14.0,  8),
]

QUERY = "database connection timeout error"


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main():
    # Build vocabulary from all texts + query
    all_texts = [t for t, _, _ in _DEMO_MEMORIES] + [QUERY]
    build_vocab(all_texts)

    retriever = MemoryRetriever()
    now = datetime.now(timezone.utc)

    entries: list[MemoryEntry] = []
    for text, days_ago, importance in _DEMO_MEMORIES:
        entry = MemoryEntry(
            text=text,
            timestamp=_dt(days_ago),
            importance_score=float(importance),
            embedding=embed(text),
        )
        retriever.add_memory(entry)
        entries.append(entry)

    print("=" * 66)
    print(f"QUERY: \"{QUERY}\"")
    print("=" * 66)

    # ---- Relevance-only retrieval ----------------------------------------
    print("\n[A] RELEVANCE-ONLY retrieval (alpha=0, beta=0, gamma=1)")
    print("-" * 66)
    top_rel = retriever.retrieve_top_k(QUERY, k=5, alpha=0, beta=0, gamma=1, now=now)
    for rank, (entry, scores) in enumerate(top_rel, 1):
        age_h = (now - entry.timestamp).total_seconds() / 3600
        print(
            f"  #{rank}  rel={scores['relevance']:.3f}  "
            f"imp={entry.importance_score:.0f}/10  age={age_h:.1f}h"
        )
        print(f"       \"{entry.text}\"")

    # ---- Hybrid retrieval -----------------------------------------------
    print("\n[B] HYBRID retrieval (alpha=1, beta=1, gamma=1)")
    print("-" * 66)
    top_hybrid = retriever.retrieve_top_k(QUERY, k=5, alpha=1, beta=1, gamma=1, now=now)
    for rank, (entry, scores) in enumerate(top_hybrid, 1):
        age_h = (now - entry.timestamp).total_seconds() / 3600
        print(
            f"  #{rank}  combined={scores['combined']:.3f}  "
            f"(rec={scores['recency']:.3f}, imp={scores['importance']:.3f}, rel={scores['relevance']:.3f})"
        )
        print(f"       \"{entry.text}\"  [imp={entry.importance_score:.0f}/10, age={age_h:.1f}h]")

    # ---- Score breakdown for all memories --------------------------------
    print("\n[C] FULL SCORE BREAKDOWN (hybrid, all memories)")
    print("-" * 66)
    print(f"  {'Text':<52} {'Rec':>5} {'Imp':>5} {'Rel':>5} {'Comb':>6}")
    print(f"  {'-'*52} {'-'*5} {'-'*5} {'-'*5} {'-'*6}")
    all_scored = retriever.retrieve_top_k(
        QUERY, k=len(_DEMO_MEMORIES), alpha=1, beta=1, gamma=1, now=now
    )
    for entry, scores in all_scored:
        text_short = entry.text[:50] + ".." if len(entry.text) > 50 else entry.text
        print(
            f"  {text_short:<52} "
            f"{scores['recency']:>5.3f} "
            f"{scores['importance']:>5.3f} "
            f"{scores['relevance']:>5.3f} "
            f"{scores['combined']:>6.3f}"
        )

    # ---- Comparison commentary ------------------------------------------
    print("\n[D] COMPARISON: relevance-only vs hybrid")
    print("-" * 66)
    rel_texts  = [e.text for e, _ in top_rel]
    hyb_texts  = [e.text for e, _ in top_hybrid]

    print("  Relevance-only top-3:")
    for i, t in enumerate(rel_texts[:3], 1):
        print(f"    {i}. {t}")

    print("  Hybrid top-3:")
    for i, t in enumerate(hyb_texts[:3], 1):
        print(f"    {i}. {t}")

    print(
        "\n  Observation: hybrid retrieval surfaces the *recent* production timeout event"
        "\n  at rank #1 because recency and importance amplify its score, while"
        "\n  relevance-only may rank older but semantically similar entries higher."
    )
    print("=" * 66)


if __name__ == "__main__":
    main()
