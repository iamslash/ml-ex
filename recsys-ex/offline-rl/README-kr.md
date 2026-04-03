# Offline RL (DQN + CQL)

> 오프라인 데이터만으로 추천 정책을 학습하는 보수적 강화학습.

## 실행 방법

```bash
cd recsys-ex/offline-rl
python offline_rl.py
```

## 핵심 개념

### 오프라인 RL의 문제

온라인 RL은 환경과 상호작용하며 학습하지만, 추천 시스템에서는 이미 수집된 로그 데이터만 사용 가능. 일반 DQN은 분포 외(out-of-distribution) 행동의 Q값을 과대추정하여 실패.

### CQL (Conservative Q-Learning)

```
L = L_TD + α * (log_sum_exp(Q(s,a)) - Q(s, a_data))
```

- `L_TD`: 일반 TD loss (Q값 학습)
- CQL 페널티: 모든 행동의 Q값을 낮추되, 데이터에 있는 행동의 Q값은 유지
- 결과: 보수적 Q값 → 안전한 정책

### 온라인 DQN과의 비교

| | Online DQN | Offline DQN + CQL |
|---|---|---|
| 데이터 | 실시간 수집 | 로그 데이터만 |
| 탐험 | epsilon-greedy | 불가능 |
| 위험 | Q값 과대추정 | CQL로 방지 |
| 배포 | 위험 | 안전 |
