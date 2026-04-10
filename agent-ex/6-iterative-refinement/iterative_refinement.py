"""
Iterative Refinement: Multi-turn refinement with quality tracking per turn.
Demo task: "Write a Python function to parse CSV with error handling"
"""

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Simulated drafts per turn
# ---------------------------------------------------------------------------

_DRAFTS = {
    1: """\
def parse_csv(filepath):
    rows = []
    with open(filepath) as f:
        for line in f:
            rows.append(line.strip().split(','))
    return rows
""",
    2: """\
def parse_csv(filepath):
    rows = []
    try:
        with open(filepath) as f:
            for line in f:
                rows.append(line.strip().split(','))
    except FileNotFoundError:
        print(f"Error: file not found: {filepath}")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
    return rows
""",
    3: """\
def parse_csv(filepath: str, delimiter: str = ',') -> list[list[str]]:
    \"\"\"
    Parse a CSV file and return rows as a list of string lists.

    Args:
        filepath: Path to the CSV file.
        delimiter: Field delimiter character (default ',').

    Returns:
        List of rows, each row is a list of field strings.
        Returns empty list on error.
    \"\"\"
    rows = []
    try:
        with open(filepath, encoding='utf-8') as f:
            for line in f:
                rows.append(line.strip().split(delimiter))
    except FileNotFoundError:
        print(f"Error: file not found: {filepath}")
        return []
    except PermissionError:
        print(f"Error: permission denied: {filepath}")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
    return rows
""",
    4: """\
import csv
from pathlib import Path


def parse_csv(
    filepath: str | Path,
    delimiter: str = ',',
    skip_empty: bool = True,
    max_rows: Optional[int] = None,
) -> list[list[str]]:
    \"\"\"
    Parse a CSV file and return rows as a list of string lists.

    Args:
        filepath: Path to the CSV file (str or Path).
        delimiter: Field delimiter character (default ',').
        skip_empty: Skip blank lines if True (default True).
        max_rows: Maximum number of rows to read (None = no limit).

    Returns:
        List of rows, each row is a list of field strings.
        Returns empty list on error.

    Raises:
        Does not raise; errors are logged and an empty list is returned.
    \"\"\"
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"Error: file not found: {filepath}")
        return []
    if not filepath.is_file():
        print(f"Error: not a file: {filepath}")
        return []

    rows: list[list[str]] = []
    try:
        with open(filepath, encoding='utf-8', newline='') as f:
            reader = csv.reader(f, delimiter=delimiter)
            for row in reader:
                if skip_empty and not any(row):
                    continue
                rows.append(row)
                if max_rows is not None and len(rows) >= max_rows:
                    break
    except PermissionError:
        print(f"Error: permission denied: {filepath}")
        return []
    except UnicodeDecodeError as e:
        print(f"Error: encoding issue in {filepath}: {e}")
        return []
    except csv.Error as e:
        print(f"Error: CSV parse error in {filepath}: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error reading {filepath}: {e}")
        return []

    return rows
""",
}

_FEEDBACKS = {
    1: (
        "Missing error handling. No try/except for FileNotFoundError. "
        "No type hints or docstring. Uses naive split instead of csv module."
    ),
    2: (
        "Error handling added. Still missing: type hints, docstring, "
        "PermissionError handling, and proper csv module usage."
    ),
    3: (
        "Type hints and docstring added. Consider: using csv.reader to handle "
        "quoted fields correctly, supporting Path objects, skip_empty option, "
        "max_rows limit, and UnicodeDecodeError handling."
    ),
    4: (
        "All major issues addressed. Uses csv.reader, Path, skip_empty, max_rows, "
        "UnicodeDecodeError, csv.Error. Quality is high."
    ),
}

# ---------------------------------------------------------------------------
# Quality Checklist (each item worth 1 point, max 10)
# ---------------------------------------------------------------------------

CHECKLIST = [
    ("error_handling",   lambda c: "try" in c and "except" in c),
    ("file_not_found",   lambda c: "FileNotFoundError" in c or "not found" in c.lower()),
    ("type_hints",       lambda c: "->" in c and ":" in c and "str" in c),
    ("docstring",        lambda c: '"""' in c or "'''" in c),
    ("csv_module",       lambda c: "import csv" in c or "csv.reader" in c),
    ("path_support",     lambda c: "Path" in c),
    ("encoding",         lambda c: "encoding" in c),
    ("unicode_error",    lambda c: "UnicodeDecodeError" in c),
    ("permission_error", lambda c: "PermissionError" in c),
    ("edge_cases",       lambda c: "skip_empty" in c or "max_rows" in c or "is_file" in c),
]


# ---------------------------------------------------------------------------
# Refinement Engine
# ---------------------------------------------------------------------------

class RefinementEngine:
    """Manages the generate -> feedback -> refine cycle."""

    def generate_draft(self, task: str) -> str:
        """Return initial (turn 1) draft."""
        return _DRAFTS[1]

    def get_feedback(self, output: str, turn: int) -> str:
        """Simulated feedback — each turn catches different issues."""
        return _FEEDBACKS.get(turn, "No major issues found.")

    def refine(self, output: str, feedback: str, turn: int) -> str:
        """Return improved output for the next turn."""
        next_draft = _DRAFTS.get(turn + 1)
        if next_draft is None:
            return output  # nothing to improve
        return next_draft

    def quality_score(self, output: str) -> float:
        """Score output against checklist (0-10)."""
        score = sum(1 for _, check in CHECKLIST if check(output))
        return float(score)

    def checklist_detail(self, output: str) -> list[tuple[str, bool]]:
        return [(name, check(output)) for name, check in CHECKLIST]


# ---------------------------------------------------------------------------
# Quality Tracker
# ---------------------------------------------------------------------------

@dataclass
class TurnRecord:
    turn: int
    output: str
    feedback: str
    score: float
    delta: float


class QualityTracker:
    """Records score per turn, detects plateau."""

    def __init__(self, epsilon: float = 0.5):
        self.epsilon = epsilon
        self.records: list[TurnRecord] = []

    def record(self, turn: int, output: str, feedback: str, score: float):
        prev_score = self.records[-1].score if self.records else 0.0
        delta = score - prev_score
        self.records.append(TurnRecord(turn, output, feedback, score, delta))

    def plateau_detected(self) -> bool:
        """Return True if the last improvement was below epsilon."""
        if len(self.records) < 2:
            return False
        return abs(self.records[-1].delta) < self.epsilon


# ---------------------------------------------------------------------------
# Main refinement loop
# ---------------------------------------------------------------------------

def run_refinement(task: str, max_turns: int = 6, epsilon: float = 0.5) -> QualityTracker:
    engine = RefinementEngine()
    tracker = QualityTracker(epsilon=epsilon)

    output = engine.generate_draft(task)

    for turn in range(1, max_turns + 1):
        score = engine.quality_score(output)
        feedback = engine.get_feedback(output, turn)
        tracker.record(turn, output, feedback, score)

        # Print turn summary
        _print_turn(tracker.records[-1], engine)

        # Plateau detection
        if turn > 1 and tracker.plateau_detected():
            print(f"\n  [STOP] Plateau detected at turn {turn} "
                  f"(delta={tracker.records[-1].delta:.1f} < epsilon={epsilon})")
            break

        # Max quality
        if score >= len(CHECKLIST):
            print(f"\n  [STOP] Maximum quality reached at turn {turn}.")
            break

        # Refine for next turn
        output = engine.refine(output, feedback, turn)

    return tracker


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _snippet(code: str, max_lines: int = 6) -> str:
    lines = code.strip().splitlines()
    if len(lines) <= max_lines:
        return "\n    ".join(lines)
    return "\n    ".join(lines[:max_lines]) + f"\n    ... ({len(lines) - max_lines} more lines)"


def _print_turn(record: TurnRecord, engine: RefinementEngine):
    print(f"\n{'='*65}")
    print(f"  Turn {record.turn}  |  Score: {record.score:.0f}/{len(CHECKLIST)}  "
          f"|  Delta: {record.delta:+.1f}")
    print('='*65)

    print(f"\n  [Code snippet]")
    print(f"    {_snippet(record.output)}")

    print(f"\n  [Feedback]")
    print(f"    {record.feedback}")

    detail = engine.checklist_detail(record.output)
    passed = [name for name, ok in detail if ok]
    failed = [name for name, ok in detail if not ok]
    print(f"\n  [Checklist] PASS: {passed}")
    if failed:
        print(f"              FAIL: {failed}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TASK = "Write a Python function to parse CSV with error handling"

    print("Iterative Refinement Demo")
    print(f"Task: {TASK}")
    print(f"Max turns: 6  |  Plateau epsilon: 0.5 points")

    tracker = run_refinement(TASK, max_turns=6, epsilon=0.5)

    print(f"\n{'='*65}")
    print("  Convergence Summary")
    print('='*65)
    print(f"  {'Turn':>4}  {'Score':>6}  {'Delta':>6}")
    print(f"  {'-'*20}")
    for rec in tracker.records:
        print(f"  {rec.turn:>4}  {rec.score:>5.0f}/{len(CHECKLIST)}  {rec.delta:>+6.1f}")

    final = tracker.records[-1]
    print(f"\n  Final score: {final.score:.0f}/{len(CHECKLIST)} "
          f"({final.score / len(CHECKLIST) * 100:.0f}%)")
