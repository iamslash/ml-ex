"""
Memory Schema Definition + SQLite Storage for Agent Experience Memory.

Implements the JSON schema from agent memory research:
  id, timestamp, task_signature, episode, diagnosis, lesson,
  patch_hint, confidence, privacy
"""

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def make_memory(
    domain: str,
    env: str,
    tools: list[str],
    goal: str,
    actions: list[str],
    outcome: str,
    error: Optional[str],
    diagnosis: str,
    lesson: str,
    patch_hint: str,
    confidence: float,
    contains_pii: bool = False,
    redacted: bool = False,
    timestamp: Optional[str] = None,
) -> dict:
    """Construct a memory record conforming to the agent memory schema."""
    return {
        "id": str(uuid.uuid4()),
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "task_signature": {
            "domain": domain,
            "env": env,
            "tools": tools,
        },
        "episode": {
            "goal": goal,
            "actions": actions,
            "outcome": outcome,
            "error": error,
        },
        "diagnosis": diagnosis,
        "lesson": lesson,
        "patch_hint": patch_hint,
        "confidence": confidence,
        "privacy": {
            "contains_pii": contains_pii,
            "redacted": redacted,
        },
    }


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------

class MemoryStore:
    """SQLite-backed store for agent experience memories."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_table()

    # ------------------------------------------------------------------
    # DDL
    # ------------------------------------------------------------------

    def _create_table(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id        TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                domain    TEXT NOT NULL,
                payload   TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_domain ON memories(domain)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ts ON memories(timestamp)"
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_memory(self, memory: dict) -> None:
        """Persist a memory record. Overwrites if id already exists."""
        self._conn.execute(
            "INSERT OR REPLACE INTO memories (id, timestamp, domain, payload) VALUES (?, ?, ?, ?)",
            (
                memory["id"],
                memory["timestamp"],
                memory["task_signature"]["domain"],
                json.dumps(memory, ensure_ascii=False),
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_all(self) -> list[dict]:
        cur = self._conn.execute(
            "SELECT payload FROM memories ORDER BY timestamp DESC"
        )
        return [json.loads(row[0]) for row in cur.fetchall()]

    def get_by_domain(self, domain: str) -> list[dict]:
        cur = self._conn.execute(
            "SELECT payload FROM memories WHERE domain = ? ORDER BY timestamp DESC",
            (domain,),
        )
        return [json.loads(row[0]) for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def delete_old(self, days: int) -> int:
        """Delete memories older than `days` days. Returns count deleted."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        cur = self._conn.execute(
            "DELETE FROM memories WHERE timestamp < ?", (cutoff,)
        )
        self._conn.commit()
        return cur.rowcount

    def count(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM memories")
        return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def print_memory(mem: dict, index: int) -> None:
    ep = mem["episode"]
    ts = mem["task_signature"]
    outcome_tag = "[SUCCESS]" if ep["outcome"] == "success" else "[FAILURE]"
    print(f"  [{index}] {outcome_tag} id={mem['id'][:8]}...")
    print(f"       domain={ts['domain']}  env={ts['env']}  tools={ts['tools']}")
    print(f"       goal:      {ep['goal']}")
    print(f"       actions:   {ep['actions']}")
    print(f"       outcome:   {ep['outcome']}")
    if ep["error"]:
        print(f"       error:     {ep['error']}")
    print(f"       diagnosis: {mem['diagnosis']}")
    print(f"       lesson:    {mem['lesson']}")
    print(f"       patch:     {mem['patch_hint']}")
    print(f"       confidence:{mem['confidence']:.2f}  pii={mem['privacy']['contains_pii']}")
    print()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    # ---- Build 5 sample memories (mix of success / failure) ---------------
    now = datetime.now(timezone.utc)

    samples = [
        make_memory(
            domain="code_execution",
            env="python3.11",
            tools=["exec", "file_write"],
            goal="Sort a list and return the median value",
            actions=["write sort function", "call median on result"],
            outcome="success",
            error=None,
            diagnosis="Algorithm correct; edge cases handled",
            lesson="Always guard against empty input before indexing",
            patch_hint="Add `if not lst: return None` at function entry",
            confidence=0.92,
            timestamp=(now - timedelta(days=1)).isoformat(),
        ),
        make_memory(
            domain="code_execution",
            env="python3.11",
            tools=["exec"],
            goal="Parse JSON from API response",
            actions=["call json.loads", "access nested key"],
            outcome="failure",
            error="KeyError: 'data'",
            diagnosis="API response schema changed; 'data' key absent in error payloads",
            lesson="Always validate top-level keys before drilling into nested structure",
            patch_hint="Use `.get('data', {})` with a fallback",
            confidence=0.85,
            timestamp=(now - timedelta(days=3)).isoformat(),
        ),
        make_memory(
            domain="web_search",
            env="browser_tool",
            tools=["search", "scrape"],
            goal="Find the capital city of a given country",
            actions=["search query", "parse first result"],
            outcome="success",
            error=None,
            diagnosis="Query formulation was precise; top result reliable",
            lesson="Prepend 'capital city of' to country name for disambiguation",
            patch_hint="query = f'capital city of {country}'",
            confidence=0.97,
            timestamp=(now - timedelta(days=7)).isoformat(),
        ),
        make_memory(
            domain="database",
            env="postgresql14",
            tools=["sql_exec", "schema_inspect"],
            goal="Insert batch of 10 000 records",
            actions=["open connection", "loop insert", "commit"],
            outcome="failure",
            error="OperationalError: server closed the connection unexpectedly",
            diagnosis="Connection timeout during long loop; no batch insert used",
            lesson="Use executemany or COPY for bulk inserts; increase statement_timeout",
            patch_hint="Replace loop with cursor.executemany(sql, rows)",
            confidence=0.78,
            timestamp=(now - timedelta(days=15)).isoformat(),
        ),
        make_memory(
            domain="web_search",
            env="browser_tool",
            tools=["search"],
            goal="Retrieve user email from search results",
            actions=["search user name", "extract email from snippet"],
            outcome="success",
            error=None,
            diagnosis="PII extracted from public search snippet",
            lesson="Redact PII before storing in shared memory",
            patch_hint="Apply regex redaction before save_memory()",
            confidence=0.65,
            contains_pii=True,
            redacted=True,
            timestamp=(now - timedelta(days=40)).isoformat(),
        ),
    ]

    store = MemoryStore(db_path=":memory:")

    print("=" * 60)
    print("STEP 1: Save 5 memories to SQLite")
    print("=" * 60)
    for mem in samples:
        store.save_memory(mem)
    print(f"  Total stored: {store.count()}\n")

    print("=" * 60)
    print("STEP 2: Retrieve ALL memories")
    print("=" * 60)
    all_mems = store.get_all()
    for i, m in enumerate(all_mems, 1):
        print_memory(m, i)

    print("=" * 60)
    print("STEP 3: Query by domain = 'web_search'")
    print("=" * 60)
    web_mems = store.get_by_domain("web_search")
    print(f"  Found {len(web_mems)} web_search memories:\n")
    for i, m in enumerate(web_mems, 1):
        print_memory(m, i)

    print("=" * 60)
    print("STEP 4: Delete memories older than 30 days")
    print("=" * 60)
    deleted = store.delete_old(days=30)
    print(f"  Deleted {deleted} old record(s). Remaining: {store.count()}\n")

    print("=" * 60)
    print("STEP 5: Remaining memories after pruning")
    print("=" * 60)
    remaining = store.get_all()
    for i, m in enumerate(remaining, 1):
        print_memory(m, i)


if __name__ == "__main__":
    main()
