# 자기 교정 에이전트 (Self-Correcting Agent)

> 메모리·검색·거버넌스·스킬 라이브러리를 통합한 캡스톤 자기 교정 에이전트.

## 실행 방법

```bash
cd agent-ex/9-self-correcting-agent
python agent.py
```

의존성: Python 표준 라이브러리만 사용 (numpy 불필요)

## 핵심 개념

### 통합 컴포넌트

| 모듈 출처 | 컴포넌트 | 역할 |
|---|---|---|
| 모듈 1 | `MemoryStore` | SQLite 기반 에피소딕 메모리 저장 |
| 모듈 3 | `Retriever` | 최신성·관련성·중요도 3축 검색 |
| 모듈 4 | `Governor` | PII 마스킹 + 안전 게이팅 |
| 모듈 8 | `SkillLibrary` | 재사용 가능한 코드 스킬 검색 |

### 에이전트 실행 흐름

```
solve(task)
  │
  ├─ Governor.is_safe() → 차단 여부 확인
  ├─ Governor.mask_pii() → PII 마스킹
  ├─ SkillLibrary.search() → 기존 스킬 활용 시도
  │
  └─ [반복: max 5회]
       ├─ _retrieve_relevant_memories() → 과거 교훈 검색
       ├─ _generate_with_context() → 코드 생성
       ├─ _evaluate() → exec 실행 + 테스트 검증
       ├─ 성공 → 반환
       └─ 실패 → _reflect_on_failure() → _store_lesson() → 재시도
```

### 어블레이션 실험

동일한 3개 태스크를 3가지 설정으로 실행하여 메모리 효과를 비교:

| 설정 | 메모리 사용 | 에피소딕 검색 |
|---|---|---|
| No memory | X | X |
| Episodic only | O | O |
| Full hybrid | O | O + 스킬 라이브러리 |

### 데모 태스크

1. **fibonacci 구현** (쉬움 - 1회 시도로 성공)
2. **중첩 배열 JSON 파싱** (중간 - 재시도 필요)
3. **LRU 캐시 구현** (어려움 - 스킬 라이브러리 활용)

## 참고 논문/자료

- **Reflexion** (Shinn et al., 2023) - 언어적 강화학습 기반 반성
  - https://arxiv.org/abs/2303.11366
- **MemGPT** (Packer et al., 2023) - 계층적 메모리 관리
  - https://arxiv.org/abs/2310.08560
- **Voyager** (Wang et al., 2023) - 스킬 라이브러리 기반 평생학습
  - https://arxiv.org/abs/2305.16291
- **LATS** (Zhou et al., 2023) - 언어 에이전트 트리 탐색
  - https://arxiv.org/abs/2310.04406
