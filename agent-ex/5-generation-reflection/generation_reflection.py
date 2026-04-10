"""
Generation-Reflection: Andrew Ng's Agentic Workflow with separate Generator
and Reflector nodes connected by a feedback loop.
"""

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class Criteria:
    completeness: float = 0.0   # 1-5
    correctness: float = 0.0    # 1-5
    clarity: float = 0.0        # 1-5

    @property
    def average(self) -> float:
        return (self.completeness + self.correctness + self.clarity) / 3.0

    def __str__(self) -> str:
        return (
            f"completeness={self.completeness:.1f}, "
            f"correctness={self.correctness:.1f}, "
            f"clarity={self.clarity:.1f}, "
            f"avg={self.average:.2f}"
        )


@dataclass
class GeneratorOutput:
    content: str
    round_num: int


@dataclass
class ReflectorOutput:
    critique: str
    scores: Criteria
    approved: bool


# ---------------------------------------------------------------------------
# Simulated LLM responses for a "Design a REST API for a todo app" task
# ---------------------------------------------------------------------------

# Round-indexed drafts — each one improves on the previous critique
_DRAFTS = {
    1: """\
REST API Design: Todo App (Draft 1)
====================================
Endpoints:
  GET    /todos        - list all todos
  POST   /todos        - create a todo
  GET    /todos/{id}   - get a single todo
  PUT    /todos/{id}   - update a todo
  DELETE /todos/{id}   - delete a todo

Request body (POST/PUT):
  { "title": "string", "done": false }

Response body:
  { "id": 1, "title": "string", "done": false }

Notes:
  - No authentication specified yet
  - No error handling defined
  - No pagination for list endpoint
""",
    2: """\
REST API Design: Todo App (Draft 2 - with error handling)
==========================================================
Endpoints:
  GET    /todos              - list todos (supports ?page=&limit=)
  POST   /todos              - create a todo
  GET    /todos/{id}         - get a single todo
  PUT    /todos/{id}         - update a todo
  DELETE /todos/{id}         - delete a todo

Request body (POST/PUT):
  { "title": "string", "done": false }

Success responses:
  200 OK         - GET, PUT
  201 Created    - POST
  204 No Content - DELETE

Error responses:
  400 Bad Request  - { "error": "title is required" }
  404 Not Found    - { "error": "todo not found" }
  500 Internal     - { "error": "internal server error" }

Pagination (GET /todos):
  Response: { "items": [...], "total": 42, "page": 1, "limit": 20 }

Notes:
  - Authentication: Bearer token via Authorization header
  - Still missing: rate limiting, versioning
""",
    3: """\
REST API Design: Todo App (Draft 3 - production-ready)
=======================================================
Base URL: /api/v1

Endpoints:
  GET    /todos              - list todos (?page=1&limit=20&done=false)
  POST   /todos              - create a todo
  GET    /todos/{id}         - get a single todo
  PUT    /todos/{id}         - full update
  PATCH  /todos/{id}         - partial update (e.g. toggle done)
  DELETE /todos/{id}         - delete a todo

Request body (POST):
  { "title": "string (required, max 255)", "done": false, "due_date": "ISO8601|null" }

Success responses:
  200 OK         - GET, PUT, PATCH
  201 Created    - POST  (includes Location header)
  204 No Content - DELETE

Error responses:
  400 Bad Request  - { "error": "validation_error", "details": [...] }
  401 Unauthorized - { "error": "authentication_required" }
  404 Not Found    - { "error": "todo_not_found" }
  429 Too Many Req - { "error": "rate_limit_exceeded", "retry_after": 60 }
  500 Internal     - { "error": "internal_server_error" }

Auth: Bearer JWT (Authorization: Bearer <token>)
Rate limit: 100 req/min per user
Versioning: /api/v1/ prefix
Pagination: cursor-based for large datasets
""",
}

# Round-indexed critiques
_CRITIQUES = {
    1: {
        "critique": (
            "Draft is missing critical production requirements:\n"
            "1. No error handling or HTTP status codes defined\n"
            "2. No pagination for the list endpoint\n"
            "3. No authentication mechanism specified\n"
            "4. PATCH endpoint missing for partial updates"
        ),
        "completeness": 2.0,
        "correctness": 3.5,
        "clarity": 3.0,
    },
    2: {
        "critique": (
            "Good progress. Remaining issues:\n"
            "1. Still missing PATCH for partial updates (e.g. toggling done)\n"
            "2. API versioning (/api/v1/) not specified\n"
            "3. Rate limiting mentioned but not specified\n"
            "4. due_date field not included in schema"
        ),
        "completeness": 3.5,
        "correctness": 4.0,
        "clarity": 4.0,
    },
    3: {
        "critique": (
            "Excellent. All major concerns addressed:\n"
            "- PATCH endpoint added\n"
            "- Versioning with /api/v1/ prefix\n"
            "- Rate limiting (429) with retry_after\n"
            "- due_date in schema\n"
            "- Cursor-based pagination noted\n"
            "API design is production-ready."
        ),
        "completeness": 4.5,
        "correctness": 4.5,
        "clarity": 4.5,
    },
}


# ---------------------------------------------------------------------------
# Generator Node
# ---------------------------------------------------------------------------

class GeneratorNode:
    """
    Takes task + optional critique context and produces a draft output.
    In a real system this would call an LLM. Here responses are simulated.
    """

    def generate(self, task: str, critique: Optional[str], round_num: int) -> GeneratorOutput:
        # Use critique to pick the next improved draft
        draft_text = _DRAFTS.get(round_num, _DRAFTS[max(_DRAFTS.keys())])
        if critique:
            prefix = f"[Incorporating critique from round {round_num - 1}]\n\n"
            draft_text = prefix + draft_text
        return GeneratorOutput(content=draft_text, round_num=round_num)


# ---------------------------------------------------------------------------
# Reflector Node
# ---------------------------------------------------------------------------

class ReflectorNode:
    """
    Takes a GeneratorOutput and evaluates it against criteria.
    Returns scores and a critique. Approves when all scores >= threshold.
    """

    def __init__(self, approval_threshold: float = 4.0):
        self.approval_threshold = approval_threshold

    def reflect(self, output: GeneratorOutput) -> ReflectorOutput:
        data = _CRITIQUES.get(output.round_num, _CRITIQUES[max(_CRITIQUES.keys())])
        scores = Criteria(
            completeness=data["completeness"],
            correctness=data["correctness"],
            clarity=data["clarity"],
        )
        approved = (
            scores.completeness >= self.approval_threshold
            and scores.correctness >= self.approval_threshold
            and scores.clarity >= self.approval_threshold
        )
        return ReflectorOutput(
            critique=data["critique"],
            scores=scores,
            approved=approved,
        )


# ---------------------------------------------------------------------------
# Feedback Loop
# ---------------------------------------------------------------------------

@dataclass
class LoopResult:
    rounds: list[dict] = field(default_factory=list)
    final_output: Optional[GeneratorOutput] = None
    approved: bool = False
    total_rounds: int = 0


class FeedbackLoop:
    """
    Generator -> Reflector -> if not approved: feed critique back -> repeat.
    """

    def __init__(
        self,
        generator: GeneratorNode,
        reflector: ReflectorNode,
        max_iterations: int = 5,
    ):
        self.generator = generator
        self.reflector = reflector
        self.max_iterations = max_iterations

    def run(self, task: str) -> LoopResult:
        result = LoopResult()
        critique: Optional[str] = None

        for i in range(1, self.max_iterations + 1):
            # Generate
            gen_output = self.generator.generate(task, critique, round_num=i)

            # Reflect
            ref_output = self.reflector.reflect(gen_output)

            round_record = {
                "round": i,
                "output": gen_output,
                "reflection": ref_output,
            }
            result.rounds.append(round_record)

            if ref_output.approved:
                result.final_output = gen_output
                result.approved = True
                result.total_rounds = i
                break

            # Feed critique back
            critique = ref_output.critique

        if not result.approved:
            result.final_output = result.rounds[-1]["output"]
            result.total_rounds = self.max_iterations

        return result


# ---------------------------------------------------------------------------
# Pretty print helpers
# ---------------------------------------------------------------------------

def separator(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print('='*65)


def print_round(record: dict):
    r = record["round"]
    out: GeneratorOutput = record["output"]
    ref: ReflectorOutput = record["reflection"]

    separator(f"Round {r}")
    print("\n[Generator Output]")
    print(out.content)

    print("[Reflector Critique]")
    print(ref.critique)

    print(f"\n[Scores]  {ref.scores}")
    status = "APPROVED" if ref.approved else "NEEDS REVISION"
    print(f"[Status]  {status}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generation-Reflection Demo")
    print("Task: Design a REST API for a todo app")
    print("Approval threshold: all criteria >= 4.0/5.0")

    generator = GeneratorNode()
    reflector = ReflectorNode(approval_threshold=4.0)
    loop = FeedbackLoop(generator, reflector, max_iterations=5)

    result = loop.run("Design a REST API for a todo app")

    for record in result.rounds:
        print_round(record)

    separator("Final Result")
    print(f"\n  Approved   : {result.approved}")
    print(f"  Total Rounds: {result.total_rounds}")
    if result.approved:
        print(f"\n  Converged in {result.total_rounds} round(s). Final scores:")
        final_ref: ReflectorOutput = result.rounds[-1]["reflection"]
        print(f"  {final_ref.scores}")
    else:
        print("  Did not converge within max iterations.")
