"""
Self-Correcting Agent (Capstone): integrates memory schema (module 1),
3-axis retrieval (module 3), memory governance (module 4), and skill
library (module 8) patterns.

Modules 2 (reflexion), 5 (generation-reflection), 6 (iterative-refinement),
and 7 (LLM-TDD) are standalone learning exercises best studied separately.
This capstone focuses on the memory + retrieval + governance + skill reuse
pipeline that ties the agent's long-term learning loop together.

All LLM responses are simulated. No external API calls. stdlib only.
"""

import sqlite3
import math
import re
import time
import json
import traceback
from dataclasses import dataclass, field
from typing import Optional


# ===========================================================================
# Module 1 pattern: SQLite episodic memory
# ===========================================================================

class MemoryStore:
    """Lightweight SQLite-backed episodic memory."""

    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS memories (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                task     TEXT,
                error    TEXT,
                lesson   TEXT,
                ts       REAL
            )"""
        )
        self.conn.commit()

    def store(self, task: str, error: str, lesson: str):
        self.conn.execute(
            "INSERT INTO memories(task, error, lesson, ts) VALUES (?,?,?,?)",
            (task, error, lesson, time.time()),
        )
        self.conn.commit()

    def fetch_all(self) -> list[dict]:
        cur = self.conn.execute("SELECT task, error, lesson FROM memories")
        return [{"task": r[0], "error": r[1], "lesson": r[2]} for r in cur.fetchall()]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]


# ===========================================================================
# Module 3 pattern: 3-axis retrieval (recency + relevance + importance)
# ===========================================================================

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+", text.lower())


def _cosine(a: dict, b: dict) -> float:
    dot = sum(a.get(t, 0.0) * b.get(t, 0.0) for t in b)
    ma = math.sqrt(sum(v * v for v in a.values())) or 1.0
    mb = math.sqrt(sum(v * v for v in b.values())) or 1.0
    return dot / (ma * mb)


def _tf(tokens: list[str]) -> dict[str, float]:
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    total = max(len(tokens), 1)
    return {t: c / total for t, c in counts.items()}


class Retriever:
    """3-axis retrieval: recency + relevance (cosine) + importance (lesson length)."""

    def retrieve(
        self,
        query: str,
        memories: list[dict],
        top_k: int = 3,
        use_episodic: bool = True,
    ) -> list[dict]:
        if not use_episodic or not memories:
            return []
        q_vec = _tf(_tokenize(query))
        scored = []
        for i, m in enumerate(memories):
            doc = f"{m['task']} {m['lesson']}"
            relevance = _cosine(q_vec, _tf(_tokenize(doc)))
            recency = (i + 1) / len(memories)          # later = higher
            importance = min(len(m["lesson"]) / 200, 1.0)
            score = 0.5 * relevance + 0.3 * recency + 0.2 * importance
            scored.append((score, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]


# ===========================================================================
# Module 4 pattern: PII masking + safety gating
# ===========================================================================

class Governor:
    PII_PATTERNS = [
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
        (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
        (re.compile(r"\b\d{16}\b"), "[CARD]"),
    ]
    BLOCKED_KEYWORDS = {"rm -rf", "DROP TABLE", "eval(", "os.system"}

    def mask_pii(self, text: str) -> str:
        for pattern, replacement in self.PII_PATTERNS:
            text = pattern.sub(replacement, text)
        return text

    def is_safe(self, text: str) -> tuple[bool, str]:
        for kw in self.BLOCKED_KEYWORDS:
            if kw in text:
                return False, f"Blocked keyword: {kw!r}"
        return True, ""


# ===========================================================================
# Module 8 pattern: Skill library (in-process, no file I/O)
# ===========================================================================

@dataclass
class Skill:
    name: str
    code: str
    tags: list[str]
    usage_count: int = 0


class SkillLibrary:
    def __init__(self):
        self._skills: dict[str, Skill] = {}

    def add(self, name: str, code: str, tags: list[str]):
        self._skills[name] = Skill(name=name, code=code, tags=tags)

    def search(self, query: str) -> Optional[Skill]:
        q_vec = _tf(_tokenize(query))
        best_score, best_skill = 0.0, None
        for skill in self._skills.values():
            doc = f"{skill.name} {' '.join(skill.tags)}"
            score = _cosine(q_vec, _tf(_tokenize(doc)))
            if score > best_score:
                best_score, best_skill = score, skill
        if best_score > 0.1 and best_skill:
            best_skill.usage_count += 1
            return best_skill
        return None

    def count(self) -> int:
        return len(self._skills)


# ===========================================================================
# Simulated code generation (no LLM API)
# ===========================================================================

# Each task maps to a sequence of (code, expected_errors_that_cause_retry)
_SIMULATED_SOLUTIONS: dict[str, list[str]] = {
    "Implement fibonacci": [
        # Attempt 1: correct immediately
        """
def fibonacci(n):
    if n <= 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
""",
    ],
    "Parse JSON with nested arrays": [
        # Attempt 1: wrong – doesn't handle list at top level
        """
import json
def parse_json(text):
    data = json.loads(text)
    return data["items"]   # KeyError if top level is a list
""",
        # Attempt 2: correct
        """
import json
def parse_json(text):
    data = json.loads(text)
    if isinstance(data, list):
        return data
    return data.get("items", data)
""",
    ],
    "Implement LRU cache": [
        # Attempt 1: missing eviction
        """
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
    def get(self, key):
        return self.cache.get(key, -1)
    def put(self, key, value):
        self.cache[key] = value   # no eviction
""",
        # Attempt 2: eviction but wrong order
        """
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.order = []
    def get(self, key):
        if key not in self.cache:
            return -1
        return self.cache[key]   # doesn't update recency
    def put(self, key, value):
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            evict = self.order.pop(0)
            del self.cache[evict]
        self.cache[key] = value
        self.order.append(key)
""",
        # Attempt 3: correct (uses OrderedDict pattern)
        """
from collections import OrderedDict
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()
    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
""",
    ],
}

# Validation tests per task
_TESTS: dict[str, str] = {
    "Implement fibonacci": """
assert fibonacci(0) == 0
assert fibonacci(1) == 1
assert fibonacci(10) == 55
""",
    "Parse JSON with nested arrays": """
import json
result = parse_json('[1, [2, 3], 4]')
assert result == [1, [2, 3], 4], f"got {result}"
result2 = parse_json('{"items": [1, 2]}')
assert result2 == [1, 2], f"got {result2}"
""",
    "Implement LRU cache": """
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
assert cache.get(1) == 1
cache.put(3, 3)           # evicts key 2
assert cache.get(2) == -1
cache.put(4, 4)           # evicts key 1
assert cache.get(1) == -1
assert cache.get(3) == 3
assert cache.get(4) == 4
""",
}

# Pre-seeded skills for the skill library
_SEED_SKILLS = [
    dict(name="lru_cache_pattern", tags=["lru", "cache", "eviction", "ordered"], code="""
from collections import OrderedDict
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()
    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
"""),
    dict(name="json_parser", tags=["json", "parse", "nested", "array"], code="""
import json
def parse_json(text):
    data = json.loads(text)
    if isinstance(data, list):
        return data
    return data.get("items", data)
"""),
]


# ===========================================================================
# SelfCorrectingAgent
# ===========================================================================

@dataclass
class SolveResult:
    task: str
    success: bool
    attempts: int
    lessons: list[str]
    final_code: str


class SelfCorrectingAgent:
    """
    Capstone self-correcting agent.

    Integrates:
      - SQLite episodic memory (module 1)
      - 3-axis memory retrieval (module 3)
      - PII masking + safety gating (module 4)
      - Voyager-style skill library (module 8)
    """

    MAX_ATTEMPTS = 5

    def __init__(self, use_memory: bool = True, use_episodic: bool = True, use_skills: bool = True):
        self.use_memory = use_memory
        self.use_episodic = use_episodic
        self.use_skills = use_skills
        self.memory_store = MemoryStore()
        self.retriever = Retriever()
        self.governor = Governor()
        self.skill_library = SkillLibrary()
        # Pre-seed skill library
        for s in _SEED_SKILLS:
            self.skill_library.add(s["name"], s["code"], s["tags"])

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def solve(self, task: str) -> SolveResult:
        print(f"\n{'='*60}")
        print(f"[Agent] Task: {task}")

        safe, reason = self.governor.is_safe(task)
        if not safe:
            print(f"[Governor] BLOCKED: {reason}")
            return SolveResult(task=task, success=False, attempts=0, lessons=[], final_code="")

        task = self.governor.mask_pii(task)
        lessons: list[str] = []

        # Try skill library first
        skill_code = self._check_skill_library(task)

        for attempt in range(self.MAX_ATTEMPTS):
            print(f"\n  [Attempt {attempt+1}]")

            # Retrieve relevant memories
            memories = self._retrieve_relevant_memories(task)
            if memories:
                print(f"  [Memory] Retrieved {len(memories)} relevant past lesson(s).")

            # Generate solution
            code = skill_code if (attempt == 0 and skill_code) else self._generate_with_context(task, memories, attempt)
            skill_code = None  # only use on first attempt

            # Evaluate
            ok, error = self._evaluate(task, code)
            if ok:
                print(f"  [Agent] SUCCESS on attempt {attempt+1}")
                return SolveResult(
                    task=task,
                    success=True,
                    attempts=attempt + 1,
                    lessons=lessons,
                    final_code=code,
                )

            print(f"  [Agent] Error: {error}")

            # Reflect and store lesson
            reflection = self._reflect_on_failure(task, error)
            print(f"  [Reflection] {reflection}")
            lessons.append(reflection)

            if self.use_memory:
                self._store_lesson(task, error, reflection)

        return SolveResult(task=task, success=False, attempts=self.MAX_ATTEMPTS, lessons=lessons, final_code="")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _retrieve_relevant_memories(self, task: str) -> list[dict]:
        if not self.use_memory:
            return []
        all_memories = self.memory_store.fetch_all()
        return self.retriever.retrieve(task, all_memories, top_k=3, use_episodic=self.use_episodic)

    def _generate_with_context(self, task: str, memories: list[dict], attempt: int) -> str:
        solutions = _SIMULATED_SOLUTIONS.get(task, ["# no solution"])
        idx = min(attempt, len(solutions) - 1)
        if memories:
            print(f"  [Generator] Using {len(memories)} memory context(s) for generation.")
        return solutions[idx]

    def _evaluate(self, task: str, code: str) -> tuple[bool, str]:
        test = _TESTS.get(task, "")
        if not test:
            return True, ""
        namespace: dict = {}
        try:
            exec(compile(code.strip(), "<code>", "exec"), namespace)
            exec(compile(test.strip(), "<test>", "exec"), namespace)
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    def _reflect_on_failure(self, task: str, error: str) -> str:
        # Simulated reflection
        if "KeyError" in error or "key" in error.lower():
            return "Lesson: check all dictionary keys before access; handle missing keys gracefully."
        if "AttributeError" in error:
            return "Lesson: verify all methods exist before calling; check object interface."
        if "evict" in task.lower() or "cache" in task.lower():
            return "Lesson: LRU cache requires tracking access order; use OrderedDict.move_to_end."
        return f"Lesson: encountered {error}; review edge cases and boundary conditions."

    def _store_lesson(self, task: str, error: str, reflection: str):
        self.memory_store.store(task, error, reflection)

    def _check_skill_library(self, task: str) -> Optional[str]:
        if not self.use_skills:
            return None
        skill = self.skill_library.search(task)
        if skill:
            print(f"  [SkillLibrary] Found matching skill: {skill.name!r}")
            return skill.code
        return None


# ===========================================================================
# Ablation evaluation
# ===========================================================================

TASKS = [
    "Implement fibonacci",
    "Parse JSON with nested arrays",
    "Implement LRU cache",
]


def run_ablation():
    print("\n" + "=" * 60)
    print("ABLATION EVALUATION")
    print("=" * 60)

    configs = [
        ("No memory",      dict(use_memory=False, use_episodic=False, use_skills=False)),
        ("Episodic only",  dict(use_memory=True,  use_episodic=True,  use_skills=False)),
        ("Full hybrid",    dict(use_memory=True,  use_episodic=True,  use_skills=True)),
    ]

    # Results table: config -> task -> SolveResult
    results: dict[str, dict[str, SolveResult]] = {}

    for config_name, kwargs in configs:
        print(f"\n{'─'*40}")
        print(f"Config: {config_name}")
        print(f"{'─'*40}")
        agent = SelfCorrectingAgent(**kwargs)
        results[config_name] = {}
        for task in TASKS:
            r = agent.solve(task)
            results[config_name][task] = r

    # Print comparison table
    print("\n\n" + "=" * 70)
    print("ABLATION COMPARISON TABLE")
    print("=" * 70)
    header = f"{'Task':<35} {'Config':<18} {'Attempts':>8} {'Success':>8} {'Lessons':>8}"
    print(header)
    print("-" * 70)
    for task in TASKS:
        for config_name, _ in configs:
            r = results[config_name][task]
            short_task = task[:33]
            print(
                f"{short_task:<35} {config_name:<18} {r.attempts:>8} "
                f"{'YES' if r.success else 'NO':>8} {len(r.lessons):>8}"
            )
        print()

    print("=" * 70)
    print("Ablation complete.")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("Self-Correcting Agent — Capstone Demo")
    print("Integrating: Memory | Retrieval | Governance | Skill Library")

    run_ablation()


if __name__ == "__main__":
    main()
