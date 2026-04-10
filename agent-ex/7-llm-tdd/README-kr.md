# LLM-TDD (테스트 주도 개발 에이전트)

> LLM이 테스트를 먼저 작성하고, 코드를 생성한 뒤, 테스트가 통과할 때까지 반복적으로 수정하는 TDD 에이전트 패턴.

## 실행 방법

```bash
cd agent-ex/7-llm-tdd
python llm_tdd.py
```

의존성: Python 표준 라이브러리만 사용 (numpy 불필요)

## 핵심 개념

| 컴포넌트 | 역할 |
|---|---|
| `TestGenerator` | 태스크 명세에서 pytest 스타일 테스트 케이스를 생성 |
| `CodeGenerator` | 태스크 + 이전 오류를 바탕으로 점진적으로 개선된 코드를 생성 |
| `TDDLoop` | 테스트 생성 → 코드 생성 → 실행 → 오류 피드백 → 반복 |

### 동작 흐름

```
[spec] → TestGenerator → [tests]
                              ↓
                        CodeGenerator (attempt 1)
                              ↓
                        _run_tests_in_sandbox
                              ↓ fail
                        CodeGenerator (attempt 2, +errors)
                              ↓ ... (반복)
                              ↓ pass
                        [SUCCESS]
```

### 데모 시나리오

- 태스크: `Stack` 클래스 (push, pop, peek, is_empty, size) 구현
- **Attempt 1**: `peek` 메서드 누락 → 실패
- **Attempt 2**: `peek`이 빈 스택에서 `None` 반환 (예외 미발생) → 실패
- **Attempt 3**: 모든 8개 테스트 통과 → 성공

### 핵심 설계 포인트

- `exec` 기반 샌드박스로 생성된 코드를 격리 실행
- `pytest.raises`를 최소 스텁으로 구현해 외부 의존성 제거
- `GenerationContext`에 이전 오류 목록을 누적해 코드 생성기에 피드백

## 참고 논문/자료

- **AlphaCode** (Li et al., 2022) - 코드 생성 및 반복 수정
- **Self-Debugging** (Chen et al., 2023) - 실행 피드백을 통한 코드 자가 수정
  - https://arxiv.org/abs/2304.05128
- **Reflexion** (Shinn et al., 2023) - 언어적 강화학습 기반 에이전트 반성
  - https://arxiv.org/abs/2303.11366
- **AgentCoder** (Huang et al., 2023) - 테스트 주도 코드 생성 에이전트
  - https://arxiv.org/abs/2312.13010
