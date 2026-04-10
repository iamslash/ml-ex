# 생성-반성 루프 (Generation-Reflection)

> Andrew Ng의 에이전틱 워크플로우: Generator와 Reflector 노드가 피드백 루프로 연결된 반복 개선 시스템

## 실행 방법

```bash
cd agent-ex/5-generation-reflection
python generation_reflection.py
```

의존성: Python 표준 라이브러리만 사용 (외부 패키지 불필요)

## 핵심 개념

| 컴포넌트 | 역할 |
|---|---|
| `GeneratorNode` | 태스크 + 이전 비평을 받아 초안 출력 생성 |
| `ReflectorNode` | 출력물을 기준(완성도/정확성/명확성)으로 평가, 점수와 비평 반환 |
| `FeedbackLoop` | Generator → Reflector → 미승인 시 비평 재주입 → 반복 |
| `Criteria` | 완성도·정확성·명확성을 각 1-5점으로 평가 |

### 루프 흐름

```
Task
  ↓
Generator ──→ draft
                ↓
            Reflector ──→ scores + critique
                ↓
         [모든 점수 ≥ 임계값?]
           ↙ YES     ↘ NO
        완료         critique → Generator (다음 라운드)
```

### 수렴 조건

- 완성도, 정확성, 명확성 모두 4.0/5.0 이상이면 승인
- 최대 N번 반복 후 미수렴 시 최선 결과 반환

## 참고 논문/자료

- [Andrew Ng - Agentic Design Patterns](https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-2-reflection/) - Reflection 패턴 원출처
- [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651) - 자기 피드백 반복 개선
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) - 언어 기반 강화학습
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) - AI 피드백 기반 품질 향상
