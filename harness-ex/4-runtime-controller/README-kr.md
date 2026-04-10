# Runtime Controller (런타임 제어기)

> LLM 에이전트의 재시도, 상태 전이, 체크포인트, 예산, 핸드오프를 한 곳에서 관리하는 런타임 오케스트레이터.

## 실행 방법

```bash
cd harness-ex/4-runtime-controller
python3 runtime_controller.py
```

의존성 없음. Python 3.8+ 표준 라이브러리만 사용.

## 핵심 개념

### 1. RetryPolicy - 재시도 정책

```
fn() 실패
   │
   ├─ attempt < max_retries?
   │      │
   │      ├─ fixed:       sleep(delay)
   │      └─ exponential: sleep(delay × 2^attempt)
   │
   └─ attempt == max_retries → RetryError
```

- `backoff="fixed"`: 매번 동일한 대기시간 (네트워크 일시 장애에 적합)
- `backoff="exponential"`: 실패할수록 대기시간 2배 증가 (rate limit 회피에 적합)
- 텔레메트리: `attempts`, `errors`, `total_time` 누적

**실패 트레이스 (Demo 1)**:
```
attempt 1 failed: LLM API timeout (call #1). Waiting 0.10s...
attempt 2 failed: LLM API timeout (call #2). Waiting 0.20s...
→ attempt 3 성공: "LLM response: plan step 1 complete"
```

### 2. StateMachine - 유한 상태 머신

```
INIT
 │ start
 ▼
PLANNING
 │ plan_ok
 ▼
EXECUTING ◄──────────────┐
 │ eval                  │ retry
 ▼                       │
EVALUATING               │
 │ pass    │ fail        │
 ▼         ▼             │
DONE    REFLECTING ───────┘
           │ give_up
           ▼
          FAILED
```

- `on_transition(callback)`: 전이마다 감사 로그 호출
- `checkpoint(state, data, file)`: JSON 파일로 현재 상태 저장
- `restore(file)`: 저장된 체크포인트에서 재개

**체크포인트 복구 트레이스 (Demo 3)**:
```
Checkpoint saved: /tmp/sm_checkpoint_demo.json (state=EVALUATING)
[!] 크래시 시뮬레이션 - 새 StateMachine 인스턴스 생성...
New machine starts at: INIT
Restored from .../sm_checkpoint_demo.json (state=EVALUATING, saved_at=...)
→ EVALUATING → REFLECTING → EXECUTING → EVALUATING → DONE
```

### 3. BudgetTracker - 토큰 예산 추적

```
LLM 호출 → track(prompt_tok, completion_tok)
              │
              ▼
         _used_tokens += total
              │
         exceeded()? ── YES ──► 호출 차단
              │
              NO
              │
         remaining() = budget - used
         estimate_cost() = used / 1000 × cost_per_1k
```

**예산 초과 트레이스 (Demo 4)**:
```
Call 1: prompt=400, completion=300 | used=700/2,000
Call 2: prompt=600, completion=400 | used=1,700/2,000
Call 3: prompt=200, completion=250 | used=2,150/2,000 → EXCEEDED
```

### 4. HandoffController - 에이전트 핸드오프

```
orchestrator
    │ handoff("orchestrator", "planner", ctx)
    ▼
  planner ──► plan 생성
    │ handoff("planner", "coder", ctx)
    ▼
  coder ──► 코드 작성 (attempt 1)
    │ handoff("coder", "reviewer", ctx)
    ▼
  reviewer ──► REJECTED (null check 누락)
    │ handoff("reviewer", "coder", ctx)
    ▼
  coder ──► 코드 수정 (attempt 2)
    │ handoff("coder", "reviewer", ctx)
    ▼
  reviewer ──► APPROVED ✓
```

- `register(name, fn)`: 에이전트 등록
- `handoff(from, to, context)`: 컨텍스트와 함께 제어 이전
- `history`: 모든 핸드오프 기록

## 출처 및 참고 자료

| 개념 | 출처 |
|------|------|
| Retry with exponential backoff | [AWS Architecture Blog - Exponential Backoff](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/) |
| State machine + persistence | [LangGraph - State persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/) |
| Agent handoffs | [OpenAI Agents SDK - Handoffs](https://openai.github.io/openai-agents-python/handoffs/) |
| Token budget tracking | [Anthropic - Token usage](https://docs.anthropic.com/en/docs/build-with-claude/token-counting) |
| Checkpoint/restore pattern | [LangGraph - Checkpointers](https://langchain-ai.github.io/langgraph/reference/checkpoints/) |
