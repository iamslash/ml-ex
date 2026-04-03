# 인과추론 (Causal Inference)

> 처리(treatment)가 결과(outcome)에 미치는 인과적 효과를 추정하는 기법들.

## 학습 순서

| # | 모델 | 디렉토리 | 핵심 개념 |
|---|------|----------|----------|
| 1 | Meta-Learners | [meta-learners](meta-learners/) | S/T/X-Learner, CATE, ATE, propensity |
| 2 | HydraNet | [hydranet](hydranet/) | 다중 처리 uplift, DR loss, 공유 표현 |

## 핵심 개념

### 잠재 결과 프레임워크 (Potential Outcomes)

```
관측: Y = T * Y(1) + (1-T) * Y(0)
CATE: tau(X) = E[Y(1) - Y(0) | X]   (조건부 평균 처리 효과)
ATE:  tau = E[Y(1) - Y(0)]           (평균 처리 효과)
```

### Meta-Learner 비교

| 방법 | 접근 | 장점 | 단점 |
|------|------|------|------|
| S-Learner | T를 특성에 포함 | 단순 | 처리 효과를 무시할 수 있음 |
| T-Learner | 처리/대조 별도 모델 | 직관적 | 데이터 효율 낮음 |
| X-Learner | 잔차 예측 + propensity 결합 | 비대칭 처리 효과에 강함 | 복잡 |
