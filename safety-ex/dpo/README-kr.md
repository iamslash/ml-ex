# DPO (Direct Preference Optimization)

> 별도 보상 모델 없이 참조 정책 대비 KL 발산을 제약으로 사용하여 선호도 데이터로 직접 정책을 최적화하는 알고리즘

## 실행 방법

```bash
cd safety-ex/dpo
python dpo.py
```

의존성: `torch`

200쌍의 합성 선호도 데이터(프롬프트 8차원 특성 벡터)로 학습한다. CPU에서도 수초 내에 완료된다.

## 핵심 개념

### DPO 손실 함수

RLHF의 목적함수를 분석적으로 풀면 보상 모델 없이 다음 손실로 표현된다.

```
L_DPO = -log(sigmoid(beta * (log pi(y_w|x)/pi_ref(y_w|x)
                           - log pi(y_l|x)/pi_ref(y_l|x))))
```

- `pi`: 학습 중인 정책 모델
- `pi_ref`: 동결된 참조 정책 (초기 정책의 복사본)
- `y_w`: 선호 응답 (chosen)
- `y_l`: 비선호 응답 (rejected)
- `beta`: KL 페널티 강도 (이 코드에서 0.1)

### 참조 정책의 역할
- 정책이 참조 정책에서 너무 멀어지는 것을 방지 (암묵적 KL 제약)
- `beta`가 클수록 참조 정책에 가까이 유지
- `beta`가 작을수록 선호도 신호에 더 집중

### Log Ratio 해석
```python
log_ratio_chosen  = p_chosen  - ref_chosen   # chosen을 얼마나 더 선호하게 됐는가
log_ratio_rejected = p_rejected - ref_rejected # rejected를 얼마나 더 선호하게 됐는가
```

손실은 `log_ratio_chosen - log_ratio_rejected`가 커지도록 유도한다. 즉, 참조 정책 대비 chosen은 더 높게, rejected는 더 낮게 채점하도록 학습된다.

### 학습 구조
- **정책 모델**: 2층 MLP (8 -> 32 -> 1)
- **참조 모델**: 정책과 동일 구조, 학습 시작 시점 가중치 복사 후 동결
- **입력**: 프롬프트 특성 벡터 (8차원)
- **출력**: 스칼라 점수 (높을수록 선호)

## 관련 모델과의 비교

| 항목 | DPO | RLHF (PPO) | SFT |
|------|-----|------------|-----|
| 보상 모델 | 불필요 | 필요 | 불필요 |
| RL 루프 | 없음 | 있음 | 없음 |
| 학습 안정성 | 높음 | 낮음 (분산 큼) | 높음 |
| 구현 복잡도 | 낮음 | 높음 | 매우 낮음 |
| 선호도 반영 | 직접 | 간접 (RM 경유) | 불가 |
| 대표 구현 | LLaMA-2-Chat | InstructGPT | GPT-3 |

### DPO vs IPO (Identity Preference Optimization)
- **DPO**: Bradley-Terry 가정 하에 유도, 극단적 선호도에서 과적합 가능
- **IPO**: Bradley-Terry 가정 없이 유도, 더 강한 정규화 효과

### DPO vs RAFT / Best-of-N
- **RAFT**: 보상 모델로 필터링한 고품질 데이터로 SFT
- **DPO**: 선호도 쌍을 직접 사용, 별도 필터링 단계 없음

### beta 하이퍼파라미터
`beta`는 탐색(선호도 학습)과 안전(참조 정책 유지) 사이의 트레이드오프를 조절한다. 실제 LLM 학습에서는 0.05~0.5 범위를 사용한다.
