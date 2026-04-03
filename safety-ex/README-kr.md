# AI 안전 (Safety)

> 모델의 안전성을 평가하고 개선하는 기법들.

## 학습 순서

| # | 모델 | 디렉토리 | 핵심 개념 |
|---|------|----------|----------|
| 1 | Toxicity Classifier | [toxicity-classifier](toxicity-classifier/) | 유해성 분류, 임계값, FPR/FNR |
| 2 | Reward Model | [reward-model](reward-model/) | Bradley-Terry, 선호도 학습 |
| 3 | DPO | [dpo](dpo/) | Direct Preference Optimization |

## 안전 파이프라인

```
데이터 수집 (인간 선호도)
  → Reward Model 학습 (reward-model)
  → 정책 최적화 (DPO 또는 RLHF)
  → 안전 필터링 (toxicity-classifier)
```
