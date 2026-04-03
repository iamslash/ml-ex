# Meta-Learners (인과추론 메타러너)

> S/T/X-Learner 비교로 처리 효과(CATE)를 추정하는 인과추론 기법들.

## 실행 방법

```bash
cd causal-ex/meta-learners
python meta_learners.py
```

## 핵심 개념

### S-Learner

처리 T를 특성에 포함하여 단일 모델로 학습. 단순하지만 처리 효과를 무시할 수 있음.

### T-Learner

처리군/대조군 각각 별도 모델 학습. 직관적이지만 데이터 효율이 낮음.

### X-Learner

1. T-Learner로 mu_0, mu_1 학습
2. 잔차(imputed treatment effect) 계산
3. 잔차를 예측하는 모델 학습
4. Propensity-weighted 결합

비대칭 처리 효과에 강하고 일반적으로 가장 정확.

### 비교

| | S-Learner | T-Learner | X-Learner |
|---|---|---|---|
| 모델 수 | 1 | 2 | 4+ |
| 복잡도 | 낮음 | 중간 | 높음 |
| 정확도 | 보통 | 보통 | 높음 |
