# Harness Engineering (하네스 엔지니어링)

> LLM 에이전트를 감싸는 제어/안전 시스템. 에이전트가 올바르게 동작하도록 통제하는 전체 실행 환경.

## 5-Layer 프레임워크

이 프레임워크는 단일 출처가 아니라 여러 산업/학술 best practice의 종합이다:

| 계층 | 역할 | 주요 출처 |
|------|------|----------|
| 프롬프트 제어 | LLM에게 뭘 해야 하는지 지시 | Anthropic system prompt hierarchy, OpenAI prompt guide |
| 입출력 검증 | LLM 출력이 올바른지 검증 | NVIDIA NeMo Guardrails, Guardrails AI, Instructor |
| 실행 환경 | LLM이 뭘 할 수 있는지 제한 | OpenAI Agents SDK tool permissions |
| 런타임 제어 | 재시도/상태/예산/핸드오프 관리 | LangGraph persistence, OpenAI handoffs |
| 사람 개입 | 위험한 행동은 사람이 승인 | OpenAI Agents SDK handoffs, LangGraph interrupt |

## 학습 순서

| # | 모듈 | 디렉토리 | 핵심 개념 |
|---|------|----------|----------|
| 1 | Prompt Template | 1-prompt-template/ | system/developer/user 계층, few-shot, 우선순위 |
| 2 | I/O Guardrails | 2-io-guardrails/ | prompt injection 탐지, JSON schema, 자동 재시도 |
| 3 | Tool Sandbox | 3-tool-sandbox/ | allowlist, 경로 차단, 위험 명령 필터 |
| 4 | Runtime Controller | 4-runtime-controller/ | 재시도, 상태 머신, 체크포인트, 예산, 핸드오프 |
| 5 | Human-in-the-Loop | 5-human-in-the-loop/ | 위험도 평가, 승인 게이트, 에스컬레이션, 감사 로그 |

## agent-ex와의 관계

agent-ex = 에이전트가 **뭘 하는가** (인지 패턴)
harness-ex = 에이전트를 **어떻게 통제하는가** (실행 환경)

agent-ex를 먼저 학습한 후 harness-ex로 넘어오는 것을 추천한다.
