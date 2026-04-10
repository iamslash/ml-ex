# 리플렉시온 (Reflexion)

> 에이전트가 실패 후 자연어로 반성문을 작성하고, 그 반성을 컨텍스트에 포함해 재시도하는 언어 강화학습 패턴

## 실행 방법

```bash
cd agent-ex/2-reflexion
python reflexion.py
```

의존성: Python 표준 라이브러리만 사용 (traceback, typing)

## 핵심 개념

### Reflexion 루프

```
생성(Generate) → 평가(Evaluate) → [실패 시] 반성(Reflect) → 재시도(Retry)
                                          ↑___________________________|
```

각 단계의 역할:

| 컴포넌트 | 역할 |
|---|---|
| `SimpleCodeAgent` | 코드 생성 (시뮬레이션: 사전 정의된 코드 반환) |
| `Evaluator` | 생성된 코드를 실행하고 테스트 케이스로 검증 |
| `Reflector` | 오류 메시지를 분석해 자연어 반성문 생성 (규칙 기반) |

### 에피소딕 메모리 버퍼

논문에서 제안한 대로 반성문은 최대 3개까지만 유지한다. 가장 오래된 반성문이 자동으로 제거되어 프롬프트 길이를 통제한다.

```
reflections = ["반성 1", "반성 2", "반성 3"]  # 최대 3개
# 새 반성 추가 시 가장 오래된 것 제거 (FIFO)
```

### 데모 시나리오

- **시도 1**: 빈 리스트에서 `IndexError` 발생
- **시도 2**: 짝수 길이 리스트에서 중앙값 계산 오류 (`off-by-one`)
- **시도 3**: 반성문 2개를 컨텍스트로 받아 올바른 코드 생성 → 성공

### Reflexion vs 단순 재시도

| 방식 | 특징 |
|---|---|
| 단순 재시도 | 동일한 오류 반복 가능 |
| Reflexion | 실패 원인을 언어로 명시화 → 다음 시도에 반영 |

## 참고 논문/자료

- Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning" (2023) https://arxiv.org/abs/2303.11366
- Madaan et al., "Self-Refine: Iterative Refinement with Self-Feedback" (2023) https://arxiv.org/abs/2303.17651
- Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (2022) https://arxiv.org/abs/2201.11903
