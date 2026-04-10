# Human-in-the-Loop (사람 개입 게이트)

> 위험도 기반 라우팅으로 LLM 에이전트의 위험 행동을 사람이 승인/거부할 수 있는 제어 계층.

## 실행 방법

```bash
cd harness-ex/5-human-in-the-loop
python3 human_in_the_loop.py
```

의존성 없음. Python 3.8+ 표준 라이브러리만 사용.
승인 큐는 `/tmp/hitl_demo_approvals.json`에 저장되며 실행 후 자동 삭제됨.

## 핵심 개념

### 1. RiskAssessor - 위험도 평가

```
action 문자열
    │
    ▼
키워드 매칭 (우선순위 순)
    │
    ├─ "drop database", "deploy to production" ──► CRITICAL
    ├─ "delete", "remove", "rm -rf"            ──► HIGH
    ├─ "execute", "run", "install"             ──► MEDIUM
    └─ "read", "view", "list"                  ──► LOW
                                                    │
                                                    └─ (default)
```

규칙 목록을 위에서 아래로 순서대로 검사하며 첫 매칭에서 중단 (short-circuit).

### 2. ApprovalGate - 승인 게이트

```
request_approval(action, risk, ctx)
    │
    ▼
pending_approvals.json 에 PENDING 레코드 기록
    │
    ▼
simulate_approval(thread_id, decision)   ← 사람/매니저가 응답
    │
    ▼
check_approval(thread_id) → APPROVED | REJECTED | TIMEOUT
```

- `input()` 사용 안 함 - JSON 파일 기반 큐로 블로킹 없이 비동기 패턴 시뮬레이션
- `thread_id`: UUID 8자리로 각 요청 추적
- `timeout_seconds`: 미응답 시 자동 거부 (Demo에서는 simulate_approval로 즉시 처리)

### 3. EscalationPolicy - 에스컬레이션 정책

```
RiskLevel
    │
    ├─ LOW      ──► 자동 승인 (로그 없음)
    │
    ├─ MEDIUM   ──► 자동 승인 + 경고 로그
    │
    ├─ HIGH     ──► 사람 승인 필요
    │                   │
    │               APPROVED / REJECTED
    │
    └─ CRITICAL ──► 사람 승인 → (승인 시) 매니저 승인
                        │               │
                     REJECTED       APPROVED / REJECTED
```

### 4. AuditTrail - 감사 로그

```
모든 결정 → log(action, risk, decision, approver, notes)
                │
                ▼
           _entries[] (메모리 내 불변 목록)
                │
                ▼
         generate_report() → 텍스트 요약
```

**실제 실행 결과 (5개 시나리오)**:

```
Action 1: "Read config file"      LOW      → auto-approved
Action 2: "Execute unit tests"    MEDIUM   → auto-approved WITH WARNING
Action 3: "Delete old log files"  HIGH     → human:operator → APPROVED
Action 4: "Deploy to production"  CRITICAL → human:operator APPROVED
                                           → human:manager  APPROVED
Action 5: "Drop database table"   CRITICAL → human:operator → REJECTED
```

**감사 리포트 출력**:
```
================================================================
  AUDIT TRAIL REPORT
================================================================
  Total decisions : 5  |  Approved: 4  |  Rejected: 1

  [OK] #01 | LOW      | APPROVED | system:auto          | Read config file
  [OK] #02 | MEDIUM   | APPROVED | system:auto-warn     | Execute unit tests
  [OK] #03 | HIGH     | APPROVED | human:operator       | Delete old log files
  [OK] #04 | CRITICAL | APPROVED | human:manager        | Deploy to production
  [!!] #05 | CRITICAL | REJECTED | human:operator       | Drop database table
================================================================
```

### 전체 흐름 아키텍처

```
Agent 행동 요청
      │
      ▼
 RiskAssessor.assess()
      │
      ▼
 EscalationPolicy.process()
      │
      ├─ LOW/MEDIUM ──────────────────────────► AuditTrail.log()
      │
      ├─ HIGH ──► ApprovalGate.request()
      │                │
      │           simulate_approval()          ← 사람 응답
      │                │
      │           check_approval() ──────────► AuditTrail.log()
      │
      └─ CRITICAL ──► ApprovalGate.request() (사람)
                           │
                      simulate_approval()      ← 사람 응답
                           │
                      APPROVED? ──► ApprovalGate.request() (매니저)
                                         │
                                    simulate_approval() ← 매니저 응답
                                         │
                                    AuditTrail.log()
```

## 출처 및 참고 자료

| 개념 | 출처 |
|------|------|
| Human-in-the-loop 패턴 | [OpenAI Agents SDK - Human in the Loop](https://openai.github.io/openai-agents-python/human_in_the_loop/) |
| LangGraph interrupt | [LangGraph - Human in the loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) |
| 위험도 분류 | [NIST AI RMF - Risk Tiers](https://airc.nist.gov/Risk) |
| 감사 로그 설계 | [OWASP - Audit Logging](https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html) |
| 승인 게이트 패턴 | [AWS Step Functions - Human approval](https://docs.aws.amazon.com/step-functions/latest/dg/tutorial-human-approval.html) |
