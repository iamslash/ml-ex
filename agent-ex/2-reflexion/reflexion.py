"""
Reflexion Pattern: error -> verbal reflection -> retry with reflection in context.

Paper: Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning" (2023)
https://arxiv.org/abs/2303.11366
"""

import traceback
from typing import Optional

# ---------------------------------------------------------------------------
# Simulated code attempts (predefined, each with a specific bug)
# ---------------------------------------------------------------------------

# Attempt 1: IndexError on empty list
_CODE_ATTEMPT_1 = '''\
def solve(lst):
    """Sort a list and return the median value."""
    lst.sort()
    mid = len(lst) // 2
    return lst[mid]  # BUG: IndexError when lst is empty
'''

# Attempt 2: off-by-one for even-length lists
_CODE_ATTEMPT_2 = '''\
def solve(lst):
    """Sort a list and return the median value."""
    if not lst:
        return None
    lst_sorted = sorted(lst)
    mid = len(lst_sorted) // 2
    # BUG: for even-length, should average lst_sorted[mid-1] and lst_sorted[mid]
    return float(lst_sorted[mid])
'''

# Attempt 3: correct implementation
_CODE_ATTEMPT_3 = '''\
def solve(lst):
    """Sort a list and return the median value."""
    if not lst:
        return None
    lst_sorted = sorted(lst)
    n = len(lst_sorted)
    mid = n // 2
    if n % 2 == 0:
        return (lst_sorted[mid - 1] + lst_sorted[mid]) / 2.0
    return float(lst_sorted[mid])
'''

_ATTEMPT_CODES = [_CODE_ATTEMPT_1, _CODE_ATTEMPT_2, _CODE_ATTEMPT_3]

# ---------------------------------------------------------------------------
# Test cases used by the evaluator
# ---------------------------------------------------------------------------

_TEST_CASES = [
    ([], None),
    ([3], 3.0),
    ([1, 2, 3], 2.0),
    ([1, 2, 3, 4], 2.5),
    ([5, 1, 9, 3], 4.0),
]

# ---------------------------------------------------------------------------
# SimpleCodeAgent
# ---------------------------------------------------------------------------

class SimpleCodeAgent:
    """Generates a code attempt (simulated; returns predefined code strings)."""

    def __init__(self):
        self._attempt_index = 0

    def generate(self, task: str, reflections: list[str]) -> str:
        """Return the next predefined code attempt, incorporating reflection context."""
        idx = min(self._attempt_index, len(_ATTEMPT_CODES) - 1)
        code = _ATTEMPT_CODES[idx]
        self._attempt_index += 1

        if reflections:
            header = "# Prior reflections:\n"
            for r in reflections:
                header += f"#   - {r}\n"
            code = header + "\n" + code

        return code

# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """Runs generated code against test cases and reports pass/fail."""

    def evaluate(self, code: str) -> tuple[bool, Optional[str]]:
        """
        Execute `code` and run _TEST_CASES against the `solve` function.
        Returns (passed: bool, error_message: str | None).
        """
        namespace: dict = {}
        try:
            exec(compile(code, "<agent_code>", "exec"), namespace)  # noqa: S102
        except SyntaxError as exc:
            return False, f"SyntaxError: {exc}"

        solve = namespace.get("solve")
        if solve is None:
            return False, "NameError: function 'solve' not defined"

        for args, expected in _TEST_CASES:
            try:
                result = solve(list(args))  # pass a copy so sort is safe
            except Exception:
                return False, traceback.format_exc().strip().splitlines()[-1]

            if result != expected:
                return (
                    False,
                    f"AssertionError: solve({args!r}) returned {result!r}, expected {expected!r}",
                )

        return True, None

# ---------------------------------------------------------------------------
# Reflector (rule-based, no LLM)
# ---------------------------------------------------------------------------

_REFLECTION_RULES = [
    ("IndexError", "The code crashes on an empty list. Guard against empty input at the start of the function."),
    ("AssertionError: solve([])", "Handle the empty-list edge case by returning None immediately."),
    ("AssertionError", "The return value is wrong for one of the test cases. Check index arithmetic and even-length averaging."),
    ("NameError", "The required function was not defined. Make sure the function is named 'solve'."),
    ("SyntaxError", "There is a syntax error in the generated code. Check for missing colons, parentheses, or indentation."),
    ("TypeError", "A type mismatch occurred. Ensure numeric operations are applied only to numbers."),
]

class Reflector:
    """Produces a natural-language reflection from an error message."""

    def reflect(self, error: str, attempt_number: int) -> str:
        for pattern, advice in _REFLECTION_RULES:
            if pattern in error:
                return f"[Attempt {attempt_number}] {advice}"
        return (
            f"[Attempt {attempt_number}] An unexpected error occurred: '{error}'. "
            "Review the full logic and edge cases."
        )

# ---------------------------------------------------------------------------
# Reflexion loop
# ---------------------------------------------------------------------------

MAX_ATTEMPTS = 3
MAX_REFLECTIONS = 3  # episodic memory buffer size (Reflexion paper)

def run_reflexion(task: str) -> None:
    agent = SimpleCodeAgent()
    evaluator = Evaluator()
    reflector = Reflector()

    reflections: list[str] = []  # bounded episodic memory buffer

    print("=" * 60)
    print(f"TASK: {task}")
    print("=" * 60)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n--- Attempt {attempt} / {MAX_ATTEMPTS} ---")

        code = agent.generate(task, reflections)
        # Print only the actual function body (strip reflection header for readability)
        func_lines = [l for l in code.splitlines() if not l.startswith("#")]
        print("Generated code:")
        print("  " + "\n  ".join(func_lines).strip())

        passed, error = evaluator.evaluate(code)

        if passed:
            print("\nResult: PASSED - all test cases correct.")
            print("\nFinal solution:")
            print("  " + "\n  ".join(func_lines).strip())
            break
        else:
            print(f"\nResult: FAILED")
            print(f"  Error: {error}")

            reflection = reflector.reflect(error, attempt)
            print(f"  Reflection: {reflection}")

            # Maintain bounded episodic memory buffer
            reflections.append(reflection)
            if len(reflections) > MAX_REFLECTIONS:
                reflections.pop(0)

            if attempt == MAX_ATTEMPTS:
                print("\nMax attempts reached. Task not solved.")

    print("\n" + "=" * 60)
    print("Episodic Reflection Buffer (final state):")
    for i, r in enumerate(reflections, 1):
        print(f"  [{i}] {r}")
    print("=" * 60)


if __name__ == "__main__":
    run_reflexion("write a function that sorts a list and returns the median")
