# I/O Guardrails (입출력 가드레일)

> LLM에 들어가는 입력을 검사하고, 나오는 출력을 검증하여 형식 위반 시 자동 재시도.

## 실행 방법

```bash
cd harness-ex/2-io-guardrails
python io_guardrails.py
```

## 핵심 개념

### 입력 가드레일 (Input)

```
사용자 입력 → Prompt Injection 탐지 → 토픽 필터 → PII 마스킹 → LLM 전달
```

- **Prompt Injection 탐지**: "ignore previous instructions", "system:" 등 패턴 차단
- **토픽 필터**: 허용 주제 외 질문 차단
- **PII 마스킹**: 이메일/전화번호를 LLM에 보내기 전 마스킹

### 출력 가드레일 (Output)

```
LLM 출력 → JSON Schema 검증 → Regex 필터 → 길이 검사 → 거부 탐지
            ↓ 실패
         에러 피드백 → LLM 재호출 (최대 N회)
```

### 자동 재시도 (Retry with Feedback)

출력이 형식을 위반하면 에러 메시지를 포함하여 재호출. NeMo Guardrails, Instructor 라이브러리의 핵심 패턴.

## 출처 및 참고 자료

- NVIDIA NeMo Guardrails: input/output/topical rails
- Guardrails AI (OSS): output validation + retry
- Instructor: Pydantic schema 기반 LLM 출력 강제
