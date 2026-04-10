# 에이전트 경험 메모리 스키마 (Agent Experience Memory Schema)

> 에이전트의 과거 경험을 구조화된 JSON 스키마로 정의하고 SQLite에 저장·조회하는 메모리 시스템

## 실행 방법

```bash
cd agent-ex/1-memory-schema
python memory_schema.py
```

의존성: Python 표준 라이브러리만 사용 (sqlite3, json, uuid, datetime)

## 핵심 개념

### 메모리 스키마 구조

| 필드 | 설명 |
|---|---|
| `id` | UUID로 생성된 고유 식별자 |
| `timestamp` | 메모리 생성 시각 (ISO 8601) |
| `task_signature` | 도메인, 실행 환경, 사용 도구 목록 |
| `episode` | 목표, 수행 액션, 결과, 오류 메시지 |
| `diagnosis` | 실패/성공 원인 분석 |
| `lesson` | 이번 경험에서 얻은 교훈 |
| `patch_hint` | 다음 시도를 위한 코드 수정 힌트 |
| `confidence` | 이 메모리의 신뢰도 (0.0 ~ 1.0) |
| `privacy` | PII 포함 여부 및 익명화 처리 여부 |

### MemoryStore (SQLite 백엔드)

- `save_memory(memory)` — 메모리 저장 (upsert)
- `get_all()` — 전체 메모리 조회 (최신순)
- `get_by_domain(domain)` — 도메인별 필터링
- `delete_old(days)` — 오래된 메모리 정리

### 왜 구조화된 스키마가 필요한가?

LLM 에이전트는 동일한 실수를 반복하는 경향이 있다. 경험을 구조화된 형태로 저장하면:
1. 유사한 작업 수행 시 과거 실패 사례를 검색해 참고할 수 있다.
2. `patch_hint`를 프롬프트에 주입해 즉각적인 개선이 가능하다.
3. `privacy` 필드로 민감 정보가 공유 메모리에 노출되는 것을 방지한다.

## 참고 논문/자료

- Wang et al., "Voyager: An Open-Ended Embodied Agent with Large Language Models" (2023) — 스킬 라이브러리와 경험 메모리 개념 https://arxiv.org/abs/2305.16291
- Zhu et al., "Ghost in the Minecraft: Generally Capable Agents for Open-World Environments" (2023) https://arxiv.org/abs/2305.17144
- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020) https://arxiv.org/abs/2005.11401
