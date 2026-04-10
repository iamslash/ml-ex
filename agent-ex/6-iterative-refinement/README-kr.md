# 반복 정제 (Iterative Refinement)

> 다중 턴 반복 정제와 턴별 품질 추적: 플래토 감지로 최적 수렴 시점 자동 결정

## 실행 방법

```bash
cd agent-ex/6-iterative-refinement
python iterative_refinement.py
```

의존성: Python 표준 라이브러리만 사용 (외부 패키지 불필요)

## 핵심 개념

| 컴포넌트 | 역할 |
|---|---|
| `RefinementEngine` | 초안 생성 → 피드백 획득 → 정제 → 품질 점수 계산 |
| `QualityTracker` | 턴별 점수 기록, 플래토(개선 < ε) 감지 |
| `CHECKLIST` | 10개 항목 체크리스트로 품질 정량화 |

### 품질 체크리스트 (각 1점, 최대 10점)

| 항목 | 설명 |
|---|---|
| error_handling | try/except 존재 여부 |
| file_not_found | FileNotFoundError 처리 |
| type_hints | 타입 힌트 사용 |
| docstring | 문서 문자열 포함 |
| csv_module | csv 모듈 사용 |
| path_support | pathlib.Path 지원 |
| encoding | 인코딩 명시 |
| unicode_error | UnicodeDecodeError 처리 |
| permission_error | PermissionError 처리 |
| edge_cases | 빈 줄 스킵, 최대 행 수 등 |

### 수렴 흐름

```
Turn 1: 기본 구현 (3/10)
  ↓ 피드백: 에러 처리 없음
Turn 2: try/except 추가 (5/10)
  ↓ 피드백: 타입 힌트·문서 없음
Turn 3: 타입 힌트 + docstring (7/10)
  ↓ 피드백: csv 모듈, Path, 엣지 케이스 필요
Turn 4: 완성형 구현 (10/10)
  ↓ 플래토 감지 → 종료
```

### 플래토 감지

```
if |score[t] - score[t-1]| < epsilon:
    stop()
```

## 참고 논문/자료

- [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651) - 반복 자기 정제
- [Andrew Ng - Agentic Design Patterns Part 1: Reflection](https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-2-reflection/) - 에이전틱 반성 패턴
- [CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing](https://arxiv.org/abs/2305.11738) - LLM 자기 수정
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) - 단계별 검증
