"""
runtime_controller.py - Runtime Orchestration for LLM Agents

Covers: retries with backoff, state machine with checkpointing,
budget tracking, and agent handoffs. All LLM calls are simulated.
"""

import json
import time
import math
import uuid
import os
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# RetryPolicy
# ---------------------------------------------------------------------------

class RetryError(Exception):
    pass


class RetryPolicy:
    """Retry a callable with fixed or exponential backoff."""

    def __init__(
        self,
        max_retries: int = 3,
        backoff: str = "exponential",
        delay_seconds: float = 1.0,
    ) -> None:
        assert backoff in ("fixed", "exponential"), "backoff must be 'fixed' or 'exponential'"
        self.max_retries = max_retries
        self.backoff = backoff
        self.delay_seconds = delay_seconds
        # Telemetry
        self.attempts: int = 0
        self.errors: List[str] = []
        self.total_time: float = 0.0

    def _wait(self, attempt: int) -> float:
        """Return sleep duration for attempt N (0-indexed)."""
        if self.backoff == "fixed":
            return self.delay_seconds
        # exponential: delay * 2^attempt
        return self.delay_seconds * (2 ** attempt)

    def execute_with_retry(
        self,
        fn: Callable[[], Any],
        error_handler: Optional[Callable[[Exception, int], None]] = None,
    ) -> Any:
        """Run fn, retrying up to max_retries times on failure."""
        start = time.monotonic()
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            self.attempts += 1
            try:
                result = fn()
                self.total_time = time.monotonic() - start
                return result
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                self.errors.append(f"attempt {attempt}: {exc}")
                if error_handler:
                    error_handler(exc, attempt)
                if attempt < self.max_retries:
                    wait = self._wait(attempt)
                    print(
                        f"  [RetryPolicy] attempt {attempt + 1} failed: {exc}. "
                        f"Waiting {wait:.2f}s before retry..."
                    )
                    time.sleep(wait)

        self.total_time = time.monotonic() - start
        raise RetryError(
            f"All {self.max_retries + 1} attempts failed. Last error: {last_exc}"
        ) from last_exc


# ---------------------------------------------------------------------------
# StateMachine
# ---------------------------------------------------------------------------

class State:
    INIT = "INIT"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    EVALUATING = "EVALUATING"
    REFLECTING = "REFLECTING"
    DONE = "DONE"
    FAILED = "FAILED"


# (current_state, event) -> next_state
DEFAULT_TRANSITIONS: Dict[Tuple[str, str], str] = {
    (State.INIT,       "start"):    State.PLANNING,
    (State.PLANNING,   "plan_ok"):  State.EXECUTING,
    (State.EXECUTING,  "eval"):     State.EVALUATING,
    (State.EVALUATING, "pass"):     State.DONE,
    (State.EVALUATING, "fail"):     State.REFLECTING,
    (State.REFLECTING, "retry"):    State.EXECUTING,
    (State.REFLECTING, "give_up"):  State.FAILED,
}


class StateMachineError(Exception):
    pass


class StateMachine:
    """Finite state machine with checkpoint/restore and transition hooks."""

    def __init__(
        self,
        transitions: Optional[Dict[Tuple[str, str], str]] = None,
    ) -> None:
        self.transitions = transitions if transitions is not None else DEFAULT_TRANSITIONS
        self.current_state: str = State.INIT
        self.data: Dict[str, Any] = {}
        self._callbacks: List[Callable[[str, str, str], None]] = []

    # -- hooks ---------------------------------------------------------------

    def on_transition(self, callback: Callable[[str, str, str], None]) -> None:
        """Register a callback(from_state, event, to_state) called on every transition."""
        self._callbacks.append(callback)

    def _fire_callbacks(self, from_state: str, event: str, to_state: str) -> None:
        for cb in self._callbacks:
            cb(from_state, event, to_state)

    # -- transitions ---------------------------------------------------------

    def send(self, event: str, data: Optional[Dict[str, Any]] = None) -> str:
        """Trigger an event; return new state."""
        key = (self.current_state, event)
        if key not in self.transitions:
            raise StateMachineError(
                f"No transition from {self.current_state!r} on event {event!r}"
            )
        next_state = self.transitions[key]
        self._fire_callbacks(self.current_state, event, next_state)
        self.current_state = next_state
        if data:
            self.data.update(data)
        return next_state

    # -- persistence ---------------------------------------------------------

    def checkpoint(self, state: str, data: Dict[str, Any], filepath: str) -> None:
        """Save state + data to a JSON checkpoint file."""
        payload = {
            "state": state,
            "data": data,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"  [StateMachine] Checkpoint saved: {filepath} (state={state})")

    def restore(self, filepath: str) -> Dict[str, Any]:
        """Load a checkpoint and return its contents. Also updates internal state."""
        with open(filepath, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        self.current_state = payload["state"]
        self.data = payload.get("data", {})
        print(
            f"  [StateMachine] Restored from {filepath} "
            f"(state={self.current_state}, saved_at={payload['saved_at']})"
        )
        return payload


# ---------------------------------------------------------------------------
# BudgetTracker
# ---------------------------------------------------------------------------

class BudgetExceededError(Exception):
    pass


class BudgetTracker:
    """Track token usage and cost for LLM calls."""

    def __init__(self, token_budget: int = 10_000, cost_per_1k_tokens: float = 0.002) -> None:
        self.token_budget = token_budget
        self.cost_per_1k_tokens = cost_per_1k_tokens
        self._used_tokens: int = 0
        self._calls: int = 0

    def track(self, prompt_tokens: int, completion_tokens: int) -> None:
        total = prompt_tokens + completion_tokens
        self._used_tokens += total
        self._calls += 1

    def remaining(self) -> int:
        return max(0, self.token_budget - self._used_tokens)

    def exceeded(self) -> bool:
        return self._used_tokens > self.token_budget

    def estimate_cost(self) -> float:
        return (self._used_tokens / 1000.0) * self.cost_per_1k_tokens

    def summary(self) -> str:
        return (
            f"calls={self._calls}, "
            f"used={self._used_tokens:,}/{self.token_budget:,} tokens, "
            f"remaining={self.remaining():,}, "
            f"cost=${self.estimate_cost():.4f}"
        )


# ---------------------------------------------------------------------------
# HandoffController
# ---------------------------------------------------------------------------

class HandoffController:
    """Transfer control between named agents, recording history."""

    def __init__(self) -> None:
        self.agents: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
        self.history: List[Dict[str, Any]] = []

    def register(self, name: str, fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        self.agents[name] = fn

    def handoff(
        self,
        from_agent: str,
        to_agent: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Transfer control from from_agent to to_agent with context."""
        if to_agent not in self.agents:
            raise KeyError(f"Agent {to_agent!r} not registered")
        record = {
            "from": from_agent,
            "to": to_agent,
            "at": datetime.now(timezone.utc).isoformat(),
            "context_keys": list(context.keys()),
        }
        self.history.append(record)
        print(f"  [Handoff] {from_agent} --> {to_agent}")
        result = self.agents[to_agent](context)
        return result

    def print_history(self) -> None:
        print("  [Handoff History]")
        for i, rec in enumerate(self.history, 1):
            print(f"    {i}. {rec['from']} -> {rec['to']} at {rec['at']}")


# ===========================================================================
# DEMO
# ===========================================================================

def _separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Demo 1: RetryPolicy - fails 2 times then succeeds
# ---------------------------------------------------------------------------

def demo_retry() -> None:
    _separator("Demo 1: RetryPolicy (exponential backoff)")

    call_count = {"n": 0}

    def flaky_llm_call() -> str:
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise ConnectionError(f"LLM API timeout (call #{call_count['n']})")
        return "LLM response: plan step 1 complete"

    def on_error(exc: Exception, attempt: int) -> None:
        print(f"  [ErrorHandler] caught: {exc} on attempt {attempt}")

    policy = RetryPolicy(max_retries=3, backoff="exponential", delay_seconds=0.1)
    result = policy.execute_with_retry(flaky_llm_call, on_error)

    print(f"\n  Result      : {result}")
    print(f"  Attempts    : {policy.attempts}")
    print(f"  Errors      : {policy.errors}")
    print(f"  Total time  : {policy.total_time:.3f}s")


# ---------------------------------------------------------------------------
# Demo 2: StateMachine walk-through
# ---------------------------------------------------------------------------

def demo_state_machine() -> None:
    _separator("Demo 2: StateMachine (INIT -> ... -> DONE)")

    sm = StateMachine()

    def audit_hook(from_s: str, event: str, to_s: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"  [{ts}] TRANSITION: {from_s} --[{event}]--> {to_s}")

    sm.on_transition(audit_hook)

    steps = [
        ("start",    {"task": "write a sorting function"}),
        ("plan_ok",  {"plan": "use quicksort"}),
        ("eval",     {"code": "def sort(a): ..."}),
        ("fail",     {"reason": "off-by-one error"}),
        ("retry",    {"fix": "adjust partition index"}),
        ("eval",     {"code": "def sort(a): ... # fixed"}),
        ("pass",     {"output": "all tests green"}),
    ]

    for event, data in steps:
        sm.send(event, data)

    print(f"\n  Final state : {sm.current_state}")
    print(f"  Accumulated data keys: {list(sm.data.keys())}")


# ---------------------------------------------------------------------------
# Demo 3: Checkpoint / crash / restore
# ---------------------------------------------------------------------------

def demo_checkpoint() -> None:
    _separator("Demo 3: Checkpoint, simulated crash, restore")

    checkpoint_file = "/tmp/sm_checkpoint_demo.json"
    sm = StateMachine()

    sm.send("start",   {"task": "analyse dataset"})
    sm.send("plan_ok", {"plan": "tokenise -> embed -> cluster"})
    sm.send("eval",    {"rows_processed": 5000})

    # Save after EVALUATING
    sm.checkpoint(sm.current_state, sm.data, checkpoint_file)

    # --- simulate crash: create a brand-new machine ---
    print("\n  [!] Simulating crash - creating new StateMachine instance...")
    sm2 = StateMachine()
    print(f"  New machine starts at: {sm2.current_state}")

    sm2.restore(checkpoint_file)
    print(f"  After restore, state = {sm2.current_state}")
    print(f"  Restored data = {sm2.data}")

    # Resume from EVALUATING -> fail path for illustration
    sm2.send("fail",    {"reason": "cluster count mismatch"})
    sm2.send("retry",   {})
    sm2.send("eval",    {"rows_processed": 5000, "clusters": 8})
    sm2.send("pass",    {"result": "clusters written to disk"})
    print(f"\n  Resumed and completed. Final state: {sm2.current_state}")

    os.remove(checkpoint_file)


# ---------------------------------------------------------------------------
# Demo 4: BudgetTracker
# ---------------------------------------------------------------------------

def demo_budget() -> None:
    _separator("Demo 4: BudgetTracker (3 LLM calls, budget depletion)")

    tracker = BudgetTracker(token_budget=2_000, cost_per_1k_tokens=0.002)

    simulated_calls = [
        (400, 300),   # call 1: 700 tokens
        (600, 400),   # call 2: 1000 tokens -> total 1700
        (200, 250),   # call 3: 450 tokens -> total 2150 -> exceeds 2000
    ]

    for i, (prompt_tok, comp_tok) in enumerate(simulated_calls, 1):
        if tracker.exceeded():
            print(f"  [BudgetTracker] Call {i} BLOCKED - budget already exceeded!")
            break
        tracker.track(prompt_tok, comp_tok)
        print(
            f"  Call {i}: prompt={prompt_tok}, completion={comp_tok} | "
            f"{tracker.summary()}"
        )
        if tracker.exceeded():
            print(f"  [BudgetTracker] Budget EXCEEDED after call {i}. Stopping.")
            break

    print(f"\n  Final: {tracker.summary()}")
    print(f"  Exceeded: {tracker.exceeded()}")


# ---------------------------------------------------------------------------
# Demo 5: HandoffController
# ---------------------------------------------------------------------------

def demo_handoff() -> None:
    _separator("Demo 5: HandoffController (planner->coder->reviewer loop)")

    controller = HandoffController()
    review_pass = {"n": 0}  # reviewer rejects first time

    def planner(ctx: Dict[str, Any]) -> Dict[str, Any]:
        print("  [Planner] Generating implementation plan...")
        return {**ctx, "plan": "1. parse input 2. transform 3. output"}

    def coder(ctx: Dict[str, Any]) -> Dict[str, Any]:
        attempt = ctx.get("coder_attempt", 0) + 1
        print(f"  [Coder] Writing code (attempt {attempt})...")
        code = "def transform(x): return x[::-1]"
        if attempt > 1:
            code += "  # fixed edge case"
        return {**ctx, "code": code, "coder_attempt": attempt}

    def reviewer(ctx: Dict[str, Any]) -> Dict[str, Any]:
        review_pass["n"] += 1
        if review_pass["n"] == 1:
            print("  [Reviewer] REJECTED: missing null check")
            return {**ctx, "review": "REJECTED", "reason": "missing null check"}
        print("  [Reviewer] APPROVED: code looks good")
        return {**ctx, "review": "APPROVED"}

    controller.register("planner",  planner)
    controller.register("coder",    coder)
    controller.register("reviewer", reviewer)

    # Orchestration loop
    ctx: Dict[str, Any] = {"task": "reverse a string safely"}
    ctx = controller.handoff("orchestrator", "planner",  ctx)
    ctx = controller.handoff("planner",      "coder",    ctx)
    ctx = controller.handoff("coder",        "reviewer", ctx)

    if ctx.get("review") == "REJECTED":
        print(f"  [Orchestrator] Review rejected ({ctx['reason']}), sending back to coder...")
        ctx = controller.handoff("reviewer", "coder",    ctx)
        ctx = controller.handoff("coder",    "reviewer", ctx)

    print(f"\n  Final review: {ctx.get('review')}")
    print(f"  Final code  : {ctx.get('code')}")
    controller.print_history()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_retry()
    demo_state_machine()
    demo_checkpoint()
    demo_budget()
    demo_handoff()
    print("\n[OK] All demos complete.")
