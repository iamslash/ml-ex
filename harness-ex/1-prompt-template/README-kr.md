# 프롬프트 템플릿 (Prompt Template)

> 시스템/개발자/사용자 메시지 계층 구조와 템플릿 관리를 시연하는 모듈

---

## 실행 방법

```bash
cd harness-ex/1-prompt-template
python prompt_template.py
```

의존성 없음 — Python 3.9+ 표준 라이브러리만 사용합니다.

---

## 핵심 개념

### 메시지 우선순위 계층

LLM 하네스에서 메시지는 고정된 우선순위 순서로 조립됩니다.
하위 역할은 상위 역할의 지시를 덮어쓸 수 없습니다.

```
┌─────────────────────────────────────────┐
│  우선순위 0  │  SYSTEM                   │  ← 최고 권한 (절대 우선)
├─────────────────────────────────────────┤
│  우선순위 1  │  DEVELOPER (CLAUDE.md)    │  ← 프로젝트 규칙 + 메모리
├─────────────────────────────────────────┤
│  (few-shot) │  USER / ASSISTANT 쌍 반복  │  ← 예시 대화
├─────────────────────────────────────────┤
│  우선순위 2  │  USER                     │  ← 실제 사용자 쿼리
└─────────────────────────────────────────┘
```

### 우선순위 충돌 해결

```
System:  "always respond in English"
User:    "한국어로 답해줘"

  ┌──────────┐     render()      ┌──────────────┐
  │ Template │ ──────────────►  │ LLM (시뮬)   │
  │          │  [SYSTEM first]  │              │
  └──────────┘                  └──────┬───────┘
                                       │
                          [SYSTEM OVERRIDE 감지]
                                       │
                                       ▼
                          "Language policy enforced:
                           responding in English."
```

### 템플릿 변수 치환

```python
template_vars = {"role": "coding", "domain": "Python"}
text = "You are a {role} assistant specialized in {domain}."
# → "You are a coding assistant specialized in Python."
```

### Few-Shot 예시 관리

```
add_few_shot() 호출 순서:

  call 1 → ~80 tokens   [OK]
  call 2 → ~240 tokens  [OK]
  call 3 → ~480 tokens  [TOKEN BUDGET WARNING: >2000 chars]
                         └─ 콘솔에 경고 출력
```

### 메모리 주입 (inject_memory)

```
retrieved_memories = [
  "User prefers concise answers",
  "User is an expert Python developer",
  "Previous: debugging asyncio timeout"
]
        │
        ▼  inject_memory()
┌─────────────────────────┐
│  DEVELOPER 섹션          │
│  project rules...        │
│  --- RETRIEVED MEMORIES -│
│  [1] User prefers ...    │
│  [2] User is expert ...  │
│  [3] Previous: ...       │
│  --- END MEMORIES ------│
└─────────────────────────┘
```

### 토큰 추정 공식

```
estimate_tokens() = (전체 문자 수) ÷ 4
(4 chars ≈ 1 token 휴리스틱 — 실제 tiktoken과 10~15% 오차 가능)
```

---

## 실패 트레이스 (Deterministic Failure)

**DEMO 4** 는 의도적 실패를 보여줍니다:

```
template_vars = {"role": "coding"}  # 'domain' 키 누락

render() 출력:
  system content: "You are a coding assistant specialized in {domain}."
                                                              ^^^^^^^^
[FAILURE] Unresolved placeholder detected
[TRACE]   PromptTemplate._fill() did not substitute {domain}.
          Root cause: template_vars missing key 'domain'.
          Fix: pass template_vars={'role': '...', 'domain': '...'}
```

---

## 출처 및 참고 자료

| 개념 | 출처 |
|------|------|
| System / Developer / User 역할 구분 | [Anthropic Claude Docs — System Prompts](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/system-prompts) |
| 메시지 우선순위 (CLAUDE.md 개념) | [Anthropic Claude Code — CLAUDE.md](https://docs.anthropic.com/en/docs/claude-code/memory) |
| Few-shot 프롬프팅 | [Anthropic Prompt Engineering Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/use-examples) |
| 토큰 추정 휴리스틱 (4 chars ≈ 1 token) | OpenAI Tokenizer 문서 기반 근사값 |
| RAG 메모리 주입 패턴 | [LangChain Memory Docs](https://python.langchain.com/docs/concepts/memory/) |

> 이 모듈의 `simulate_llm()` 함수는 실제 LLM API를 호출하지 않습니다.
> 우선순위 충돌 동작은 시스템 메시지 내용을 검사하는 결정론적 로직으로 시뮬레이션합니다.
