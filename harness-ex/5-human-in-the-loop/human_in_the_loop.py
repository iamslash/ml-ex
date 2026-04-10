"""
human_in_the_loop.py - Human Approval Gates with Risk-Based Routing

Covers: risk assessment, approval gates (JSON-file queue, no blocking input()),
escalation policy, and audit trail. All approvals are simulated deterministically.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Risk levels
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# RiskAssessor
# ---------------------------------------------------------------------------

class RiskAssessor:
    """Classify the risk level of an action based on keywords and context."""

    # Priority order matters: checked top-to-bottom, first match wins.
    _RULES: List[tuple] = [
        (RiskLevel.CRITICAL, ["deploy to production", "production deploy", "drop database",
                               "drop table", "truncate table", "wipe database"]),
        (RiskLevel.HIGH,     ["delete", "remove", "rm -rf", "format", "overwrite"]),
        (RiskLevel.MEDIUM,   ["execute", "run", "install", "update", "write"]),
        (RiskLevel.LOW,      ["read", "view", "list", "get", "fetch", "describe"]),
    ]

    def assess(self, action: str, context: Optional[Dict[str, Any]] = None) -> RiskLevel:
        """Return RiskLevel for the given action string."""
        action_lower = action.lower()
        for level, keywords in self._RULES:
            if any(kw in action_lower for kw in keywords):
                return level
        return RiskLevel.LOW   # default safe


# ---------------------------------------------------------------------------
# ApprovalGate
# ---------------------------------------------------------------------------

class ApprovalStatus(str, Enum):
    PENDING  = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    TIMEOUT  = "TIMEOUT"


class ApprovalGate:
    """
    Manages approval requests via a JSON file queue.
    Does NOT call input() - human responses are injected via simulate_approval().
    """

    def __init__(
        self,
        queue_file: str = "/tmp/pending_approvals.json",
        timeout_seconds: float = 30.0,
    ) -> None:
        self.queue_file = queue_file
        self.timeout_seconds = timeout_seconds
        self._ensure_file()

    # -- internal helpers ----------------------------------------------------

    def _ensure_file(self) -> None:
        if not os.path.exists(self.queue_file):
            with open(self.queue_file, "w") as fh:
                json.dump({}, fh)

    def _load(self) -> Dict[str, Any]:
        with open(self.queue_file, "r") as fh:
            return json.load(fh)

    def _save(self, data: Dict[str, Any]) -> None:
        with open(self.queue_file, "w") as fh:
            json.dump(data, fh, indent=2)

    # -- public API ----------------------------------------------------------

    def request_approval(
        self,
        action: str,
        risk: RiskLevel,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Write an approval request. Returns thread_id."""
        thread_id = str(uuid.uuid4())[:8]
        record = {
            "thread_id":   thread_id,
            "action":      action,
            "risk":        risk.value,
            "context":     context or {},
            "status":      ApprovalStatus.PENDING.value,
            "requested_at": datetime.now(timezone.utc).isoformat(),
            "decided_at":  None,
            "decided_by":  None,
        }
        data = self._load()
        data[thread_id] = record
        self._save(data)
        print(f"  [ApprovalGate] Request written (thread={thread_id}, risk={risk.value}): {action!r}")
        return thread_id

    def check_approval(self, thread_id: str) -> ApprovalStatus:
        """Return current status for the given thread_id."""
        data = self._load()
        if thread_id not in data:
            raise KeyError(f"Unknown thread_id: {thread_id!r}")
        return ApprovalStatus(data[thread_id]["status"])

    def simulate_approval(self, thread_id: str, decision: ApprovalStatus, approver: str = "human") -> None:
        """Simulate a human (or manager) responding to an approval request."""
        data = self._load()
        if thread_id not in data:
            raise KeyError(f"Unknown thread_id: {thread_id!r}")
        data[thread_id]["status"]     = decision.value
        data[thread_id]["decided_at"] = datetime.now(timezone.utc).isoformat()
        data[thread_id]["decided_by"] = approver
        self._save(data)
        print(f"  [ApprovalGate] Decision recorded (thread={thread_id}): {decision.value} by {approver!r}")

    def cleanup(self) -> None:
        if os.path.exists(self.queue_file):
            os.remove(self.queue_file)


# ---------------------------------------------------------------------------
# EscalationPolicy
# ---------------------------------------------------------------------------

class EscalationPolicy:
    """
    Route actions through the correct approval channel based on risk level.

    LOW      -> auto-approve silently
    MEDIUM   -> auto-approve with warning log
    HIGH     -> require human approval
    CRITICAL -> require human approval + manager approval
    """

    def __init__(self, gate: ApprovalGate, audit: "AuditTrail") -> None:
        self.gate  = gate
        self.audit = audit

    def process(
        self,
        action: str,
        risk: RiskLevel,
        context: Optional[Dict[str, Any]] = None,
        # Callbacks for simulating human/manager decisions
        human_decision: ApprovalStatus  = ApprovalStatus.APPROVED,
        manager_decision: ApprovalStatus = ApprovalStatus.APPROVED,
    ) -> ApprovalStatus:
        """Execute the escalation path and return the final decision."""

        if risk == RiskLevel.LOW:
            print(f"  [Escalation] LOW risk -> auto-approved: {action!r}")
            self.audit.log(action, risk, ApprovalStatus.APPROVED, approver="system:auto")
            return ApprovalStatus.APPROVED

        if risk == RiskLevel.MEDIUM:
            print(f"  [Escalation] MEDIUM risk -> auto-approved WITH WARNING: {action!r}")
            self.audit.log(action, risk, ApprovalStatus.APPROVED, approver="system:auto-warn")
            return ApprovalStatus.APPROVED

        if risk == RiskLevel.HIGH:
            tid = self.gate.request_approval(action, risk, context)
            self.gate.simulate_approval(tid, human_decision, approver="human:operator")
            status = self.gate.check_approval(tid)
            self.audit.log(action, risk, status, approver="human:operator")
            return status

        # CRITICAL: human + manager
        if risk == RiskLevel.CRITICAL:
            tid1 = self.gate.request_approval(action, risk, context)
            self.gate.simulate_approval(tid1, human_decision, approver="human:operator")
            human_status = self.gate.check_approval(tid1)

            if human_status != ApprovalStatus.APPROVED:
                self.audit.log(action, risk, human_status, approver="human:operator")
                return human_status

            # Second approval: manager
            tid2 = self.gate.request_approval(f"[MANAGER GATE] {action}", risk, context)
            self.gate.simulate_approval(tid2, manager_decision, approver="human:manager")
            manager_status = self.gate.check_approval(tid2)
            self.audit.log(
                action, risk, manager_status,
                approver="human:manager",
                notes=f"human_tid={tid1}, manager_tid={tid2}"
            )
            return manager_status

        raise ValueError(f"Unhandled risk level: {risk}")


# ---------------------------------------------------------------------------
# AuditTrail
# ---------------------------------------------------------------------------

class AuditTrail:
    """Immutable log of every approval decision."""

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []

    def log(
        self,
        action: str,
        risk: RiskLevel,
        decision: ApprovalStatus,
        approver: str,
        notes: str = "",
    ) -> None:
        entry = {
            "seq":      len(self._entries) + 1,
            "at":       datetime.now(timezone.utc).isoformat(),
            "action":   action,
            "risk":     risk.value,
            "decision": decision.value,
            "approver": approver,
            "notes":    notes,
        }
        self._entries.append(entry)

    def generate_report(self) -> str:
        lines = [
            "",
            "=" * 64,
            "  AUDIT TRAIL REPORT",
            f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "=" * 64,
        ]
        approved = sum(1 for e in self._entries if e["decision"] == ApprovalStatus.APPROVED)
        rejected = sum(1 for e in self._entries if e["decision"] == ApprovalStatus.REJECTED)

        lines.append(f"  Total decisions : {len(self._entries)}")
        lines.append(f"  Approved        : {approved}")
        lines.append(f"  Rejected        : {rejected}")
        lines.append("")

        for e in self._entries:
            icon = "OK" if e["decision"] == ApprovalStatus.APPROVED else "!!"
            lines.append(
                f"  [{icon}] #{e['seq']:02d} | {e['risk']:<8} | {e['decision']:<8} | "
                f"{e['approver']:<20} | {e['action']}"
            )
            if e["notes"]:
                lines.append(f"           notes: {e['notes']}")

        lines.append("=" * 64)
        return "\n".join(lines)


# ===========================================================================
# DEMO
# ===========================================================================

def _separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def demo_human_in_the_loop() -> None:
    _separator("Human-in-the-Loop Demo")

    assessor = RiskAssessor()
    audit    = AuditTrail()
    gate     = ApprovalGate(queue_file="/tmp/hitl_demo_approvals.json")
    policy   = EscalationPolicy(gate, audit)

    # 5 actions with predetermined human/manager decisions
    scenarios = [
        # (action, context, human_decision, manager_decision)
        (
            "Read config file",
            {"path": "/etc/app/config.yaml"},
            ApprovalStatus.APPROVED,
            ApprovalStatus.APPROVED,
        ),
        (
            "Execute unit tests",
            {"command": "pytest tests/"},
            ApprovalStatus.APPROVED,
            ApprovalStatus.APPROVED,
        ),
        (
            "Delete old log files",
            {"path": "/var/log/app/*.log", "older_than_days": 30},
            ApprovalStatus.APPROVED,   # human approves
            ApprovalStatus.APPROVED,
        ),
        (
            "Deploy to production",
            {"service": "api-gateway", "version": "v2.3.1", "region": "us-east-1"},
            ApprovalStatus.APPROVED,   # human approves
            ApprovalStatus.APPROVED,   # manager approves
        ),
        (
            "Drop database table",
            {"table": "user_sessions", "db": "prod-db"},
            ApprovalStatus.REJECTED,   # human rejects
            ApprovalStatus.APPROVED,
        ),
    ]

    for i, (action, context, human_dec, manager_dec) in enumerate(scenarios, 1):
        print(f"\n--- Action {i}: {action!r} ---")
        risk = assessor.assess(action, context)
        print(f"  Risk assessed : {risk.value}")

        final = policy.process(
            action,
            risk,
            context,
            human_decision=human_dec,
            manager_decision=manager_dec,
        )
        print(f"  Final outcome : {final.value}")

    # Print full audit report
    print(audit.generate_report())

    # Cleanup
    gate.cleanup()


if __name__ == "__main__":
    demo_human_in_the_loop()
    print("\n[OK] Demo complete.")
