"""
Voyager-style Skill Library: store, search, and reuse successful code snippets.

Pattern: Persistent skill memory with bag-of-words similarity search,
         test verification, and skill composition.
"""

import math
import re
import traceback
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Skill dataclass
# ---------------------------------------------------------------------------

@dataclass
class Skill:
    name: str
    description: str
    code: str
    test_code: str
    tags: list[str]
    usage_count: int = 0
    success_rate: float = 1.0   # 0.0 – 1.0
    _verify_runs: int = 0
    _verify_passes: int = 0

    def summary(self) -> str:
        return (
            f"Skill({self.name!r}, tags={self.tags}, "
            f"usage={self.usage_count}, success_rate={self.success_rate:.2f})"
        )


# ---------------------------------------------------------------------------
# Bag-of-words similarity
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+", text.lower())


def _tf(tokens: list[str]) -> dict[str, float]:
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    total = max(len(tokens), 1)
    return {t: c / total for t, c in counts.items()}


def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    dot = sum(a.get(t, 0.0) * b.get(t, 0.0) for t in b)
    mag_a = math.sqrt(sum(v * v for v in a.values())) or 1.0
    mag_b = math.sqrt(sum(v * v for v in b.values())) or 1.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# SkillLibrary
# ---------------------------------------------------------------------------

class SkillLibrary:
    """
    Voyager-inspired skill library.

    Public API:
        add_skill(name, description, code, test_code, tags)
        search_skills(query, top_k) -> list[Skill]
        get_skill(name) -> Skill
        verify_skill(name) -> bool
        compose_skills(skill_names) -> Skill
    """

    def __init__(self):
        self._skills: dict[str, Skill] = {}

    # ------------------------------------------------------------------
    # Add
    # ------------------------------------------------------------------

    def add_skill(
        self,
        name: str,
        description: str,
        code: str,
        test_code: str,
        tags: list[str],
    ) -> Skill:
        skill = Skill(
            name=name,
            description=description,
            code=code,
            test_code=test_code,
            tags=tags,
        )
        self._skills[name] = skill
        print(f"[SkillLibrary] Added skill: {name!r}")
        return skill

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_skills(self, query: str, top_k: int = 3) -> list[Skill]:
        query_tf = _tf(_tokenize(query))
        scored: list[tuple[float, Skill]] = []
        for skill in self._skills.values():
            doc = f"{skill.description} {' '.join(skill.tags)}"
            skill_tf = _tf(_tokenize(doc))
            sim = _cosine_similarity(query_tf, skill_tf)
            scored.append((sim, skill))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [s for _, s in scored[:top_k]]
        for s in results:
            s.usage_count += 1
        return results

    # ------------------------------------------------------------------
    # Get
    # ------------------------------------------------------------------

    def get_skill(self, name: str) -> Skill:
        if name not in self._skills:
            raise KeyError(f"Skill not found: {name!r}")
        skill = self._skills[name]
        skill.usage_count += 1
        return skill

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------

    def verify_skill(self, name: str) -> bool:
        skill = self._skills[name]
        skill._verify_runs += 1
        namespace: dict = {}
        try:
            exec(compile(skill.code, "<skill_code>", "exec"), namespace)
            exec(compile(skill.test_code, "<skill_tests>", "exec"), namespace)
            # Run all test_ functions found
            test_fns = {k: v for k, v in namespace.items() if callable(v) and k.startswith("test_")}
            for fn in test_fns.values():
                fn()
            skill._verify_passes += 1
            skill.success_rate = skill._verify_passes / skill._verify_runs
            print(f"[verify] {name!r}: PASS  (success_rate={skill.success_rate:.2f})")
            return True
        except Exception as e:
            skill.success_rate = skill._verify_passes / skill._verify_runs
            print(f"[verify] {name!r}: FAIL  ({type(e).__name__}: {e})")
            return False

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose_skills(self, skill_names: list[str], composed_name: Optional[str] = None) -> Skill:
        parts = [self._skills[n] for n in skill_names]
        name = composed_name or "_".join(skill_names)
        description = "Composed skill: " + " + ".join(p.description for p in parts)
        code = "\n\n".join(
            f"# ---- {p.name} ----\n{p.code.strip()}" for p in parts
        )
        test_code = "\n\n".join(
            f"# ---- tests for {p.name} ----\n{p.test_code.strip()}" for p in parts
        )
        tags = list({tag for p in parts for tag in p.tags})
        composed = self.add_skill(name, description, code, test_code, tags)
        print(f"[SkillLibrary] Composed {skill_names} -> {name!r}")
        return composed

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        skills = list(self._skills.values())
        if not skills:
            return {"total": 0, "avg_success_rate": 0.0, "total_usages": 0}
        return {
            "total": len(skills),
            "avg_success_rate": sum(s.success_rate for s in skills) / len(skills),
            "total_usages": sum(s.usage_count for s in skills),
        }


# ---------------------------------------------------------------------------
# Predefined skills for the demo
# ---------------------------------------------------------------------------

SKILLS = {
    "sort_list": dict(
        description="Sort a list of elements in ascending order",
        tags=["sort", "list", "ordering"],
        code="""
def sort_list(lst):
    return sorted(lst)
""",
        test_code="""
def test_sort_list():
    assert sort_list([3, 1, 2]) == [1, 2, 3]
    assert sort_list([]) == []
    assert sort_list([1]) == [1]
""",
    ),
    "binary_search": dict(
        description="Find element index in sorted array using binary search",
        tags=["search", "binary", "sorted", "array", "element", "find"],
        code="""
def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
""",
        test_code="""
def test_binary_search_found():
    assert binary_search([1, 3, 5, 7, 9], 5) == 2

def test_binary_search_not_found():
    assert binary_search([1, 3, 5], 4) == -1

def test_binary_search_empty():
    assert binary_search([], 1) == -1
""",
    ),
    "parse_csv": dict(
        description="Parse CSV text into list of dictionaries using header row",
        tags=["csv", "parse", "text", "data"],
        code="""
def parse_csv(text):
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return []
    headers = [h.strip() for h in lines[0].split(',')]
    rows = []
    for line in lines[1:]:
        values = [v.strip() for v in line.split(',')]
        rows.append(dict(zip(headers, values)))
    return rows
""",
        test_code="""
def test_parse_csv_basic():
    csv = "name,age\\nAlice,30\\nBob,25"
    result = parse_csv(csv)
    assert result == [{'name': 'Alice', 'age': '30'}, {'name': 'Bob', 'age': '25'}]

def test_parse_csv_empty():
    assert parse_csv("") == []
""",
    ),
    "retry_with_backoff": dict(
        description="Retry a function call with exponential backoff on exception",
        tags=["retry", "backoff", "resilience", "error"],
        code="""
import time

def retry_with_backoff(fn, max_retries=3, base_delay=0.01):
    delay = base_delay
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    return fn()
""",
        test_code="""
def test_retry_success_first():
    calls = []
    def fn():
        calls.append(1)
        return 42
    assert retry_with_backoff(fn) == 42
    assert len(calls) == 1

def test_retry_eventual_success():
    calls = []
    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise ValueError("not yet")
        return "ok"
    assert retry_with_backoff(fn, max_retries=3, base_delay=0.001) == "ok"
    assert len(calls) == 3
""",
    ),
    "validate_email": dict(
        description="Validate whether a string is a well-formed email address",
        tags=["email", "validate", "regex", "string"],
        code="""
import re

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+\\-]+@[a-zA-Z0-9.\\-]+\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))
""",
        test_code="""
def test_valid_email():
    assert validate_email("user@example.com") is True

def test_invalid_email_no_at():
    assert validate_email("userexample.com") is False

def test_invalid_email_no_domain():
    assert validate_email("user@") is False
""",
    ),
}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Voyager-style Skill Library Demo")
    print("=" * 60)

    lib = SkillLibrary()

    # 1. Add skills
    print("\n--- Adding Skills ---")
    for name, meta in SKILLS.items():
        lib.add_skill(
            name=name,
            description=meta["description"],
            code=meta["code"],
            test_code=meta["test_code"],
            tags=meta["tags"],
        )

    # 2. Search
    print("\n--- Search: 'find element in sorted array' ---")
    results = lib.search_skills("find element in sorted array", top_k=3)
    for i, skill in enumerate(results, 1):
        print(f"  {i}. {skill.summary()}")

    # 3. Exact retrieval
    print("\n--- Get skill: 'parse_csv' ---")
    csv_skill = lib.get_skill("parse_csv")
    print(f"  {csv_skill.summary()}")

    # 4. Verify skills
    print("\n--- Verifying all skills ---")
    for name in SKILLS:
        lib.verify_skill(name)

    # 5. Compose
    print("\n--- Composing 'sort_list' + 'binary_search' -> 'sorted_search' ---")
    composed = lib.compose_skills(["sort_list", "binary_search"], composed_name="sorted_search")
    lib.verify_skill("sorted_search")

    # 6. Stats
    print("\n--- Library Stats ---")
    s = lib.stats()
    print(f"  Total skills    : {s['total']}")
    print(f"  Avg success rate: {s['avg_success_rate']:.2f}")
    print(f"  Total usages    : {s['total_usages']}")


if __name__ == "__main__":
    main()
