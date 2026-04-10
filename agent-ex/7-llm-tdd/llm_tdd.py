"""
LLM-TDD: Agent writes tests first, then code, then iterates until tests pass.

Pattern: Test-Driven Development with LLM simulation
- TestGenerator: generates pytest-style test cases from a task spec
- CodeGenerator: generates progressively better code on each attempt
- TDDLoop: orchestrates the TDD cycle
"""

import sys
import traceback
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Simulated LLM responses (no API calls)
# ---------------------------------------------------------------------------

GENERATED_TESTS = '''
# pytest is injected by the sandbox runner

def test_push_and_size():
    s = Stack()
    s.push(1)
    s.push(2)
    assert s.size() == 2

def test_pop_returns_last():
    s = Stack()
    s.push(10)
    s.push(20)
    assert s.pop() == 20
    assert s.size() == 1

def test_pop_empty_raises():
    s = Stack()
    with pytest.raises(IndexError):
        s.pop()

def test_peek_returns_top():
    s = Stack()
    s.push(5)
    s.push(9)
    assert s.peek() == 9
    assert s.size() == 2  # peek must not remove element

def test_peek_empty_raises():
    s = Stack()
    with pytest.raises(IndexError):
        s.peek()

def test_is_empty_true():
    s = Stack()
    assert s.is_empty() is True

def test_is_empty_false():
    s = Stack()
    s.push(42)
    assert s.is_empty() is False

def test_multiple_push_pop():
    s = Stack()
    for i in range(5):
        s.push(i)
    results = []
    while not s.is_empty():
        results.append(s.pop())
    assert results == [4, 3, 2, 1, 0]
'''

# Attempt 1: missing peek method
CODE_ATTEMPT_1 = '''
class Stack:
    def __init__(self):
        self._data = []

    def push(self, item):
        self._data.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._data.pop()

    def is_empty(self):
        return len(self._data) == 0

    def size(self):
        return len(self._data)
    # NOTE: peek is missing
'''

# Attempt 2: peek exists but doesn't raise on empty
CODE_ATTEMPT_2 = '''
class Stack:
    def __init__(self):
        self._data = []

    def push(self, item):
        self._data.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._data.pop()

    def peek(self):
        # BUG: returns None on empty instead of raising
        if self.is_empty():
            return None
        return self._data[-1]

    def is_empty(self):
        return len(self._data) == 0

    def size(self):
        return len(self._data)
'''

# Attempt 3: fully correct implementation
CODE_ATTEMPT_3 = '''
class Stack:
    def __init__(self):
        self._data = []

    def push(self, item):
        self._data.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._data.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self._data[-1]

    def is_empty(self):
        return len(self._data) == 0

    def size(self):
        return len(self._data)
'''

ATTEMPTS = [CODE_ATTEMPT_1, CODE_ATTEMPT_2, CODE_ATTEMPT_3]


# ---------------------------------------------------------------------------
# Core classes
# ---------------------------------------------------------------------------

class TestGenerator:
    """Generates test cases from a task specification (simulated LLM)."""

    def generate(self, spec: str) -> str:
        print(f"[TestGenerator] Generating tests for: '{spec}'")
        return GENERATED_TESTS


@dataclass
class GenerationContext:
    task: str
    previous_errors: list[str] = field(default_factory=list)
    attempt: int = 0


class CodeGenerator:
    """Generates implementation code, improving on each attempt (simulated LLM)."""

    def generate(self, ctx: GenerationContext) -> str:
        idx = min(ctx.attempt, len(ATTEMPTS) - 1)
        attempt_label = ctx.attempt + 1
        if ctx.previous_errors:
            print(f"[CodeGenerator] Attempt {attempt_label}: regenerating with {len(ctx.previous_errors)} previous error(s).")
        else:
            print(f"[CodeGenerator] Attempt {attempt_label}: generating initial code.")
        return ATTEMPTS[idx]


@dataclass
class TestResult:
    passed: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def all_pass(self) -> bool:
        return self.failed == 0 and len(self.errors) == 0


def _run_tests_in_sandbox(test_code: str, impl_code: str) -> TestResult:
    """Execute tests in an isolated namespace. Uses exec (sandboxed via local scope)."""
    namespace: dict = {}

    # Execute implementation
    try:
        exec(compile(impl_code, "<impl>", "exec"), namespace)
    except Exception as e:
        return TestResult(failed=1, errors=[f"Implementation compile error: {e}"])

    # Collect test functions
    test_functions = {
        name: fn
        for name, fn in namespace.items()
        if callable(fn) and name.startswith("test_")
    }

    # We need pytest.raises — provide a minimal stub
    import contextlib

    class _RaisesContext:
        def __init__(self, exc_type):
            self.exc_type = exc_type

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                raise AssertionError(f"Expected {self.exc_type.__name__} but no exception raised")
            if not issubclass(exc_type, self.exc_type):
                raise AssertionError(
                    f"Expected {self.exc_type.__name__} but got {exc_type.__name__}"
                )
            return True  # suppress the expected exception

    class _Pytest:
        @staticmethod
        def raises(exc_type):
            return _RaisesContext(exc_type)

    # Inject the test helpers into namespace
    _pytest_stub = _Pytest()
    full_ns = {**namespace, "pytest": _pytest_stub}
    exec(compile(test_code, "<tests>", "exec"), full_ns)

    test_fns = {k: v for k, v in full_ns.items() if callable(v) and k.startswith("test_")}

    result = TestResult()
    for name, fn in test_fns.items():
        try:
            fn()
            result.passed += 1
        except Exception as e:
            result.failed += 1
            result.errors.append(f"{name}: {type(e).__name__}: {e}")

    return result


class TDDLoop:
    """
    Orchestrates the TDD cycle:
      1. Generate tests from spec
      2. Generate code
      3. Run tests in sandbox
      4. If fail: collect errors, feed back to CodeGenerator
      5. Repeat until all pass or max_attempts
    """

    def __init__(self, max_attempts: int = 5):
        self.max_attempts = max_attempts
        self.test_gen = TestGenerator()
        self.code_gen = CodeGenerator()

    def run(self, spec: str) -> bool:
        print("=" * 60)
        print(f"TDD Loop starting for spec: '{spec}'")
        print("=" * 60)

        test_code = self.test_gen.generate(spec)
        print("\n--- Generated Tests ---")
        print(test_code)

        ctx = GenerationContext(task=spec)
        previous_errors: list[str] = []

        for attempt in range(self.max_attempts):
            ctx.attempt = attempt
            ctx.previous_errors = previous_errors

            impl_code = self.code_gen.generate(ctx)
            print(f"\n--- Attempt {attempt + 1} Code ---")
            print(impl_code)

            result = _run_tests_in_sandbox(test_code, impl_code)

            print(f"\n--- Attempt {attempt + 1} Results ---")
            print(f"  Passed : {result.passed}")
            print(f"  Failed : {result.failed}")
            if result.errors:
                for err in result.errors:
                    print(f"  ERROR  : {err}")

            if result.all_pass:
                print(f"\n[TDDLoop] All tests passed on attempt {attempt + 1}!")
                return True

            previous_errors = result.errors

        print(f"\n[TDDLoop] Failed after {self.max_attempts} attempts.")
        return False


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    spec = "Implement a Stack class with push, pop, peek, is_empty, size"
    loop = TDDLoop(max_attempts=5)
    success = loop.run(spec)
    print("\n" + "=" * 60)
    print(f"Final outcome: {'SUCCESS' if success else 'FAILURE'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
