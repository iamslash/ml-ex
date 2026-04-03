# Offline RL with DQN + CQL (오프라인 강화학습)

> 환경과의 상호작용 없이 수집된 오프라인 데이터만으로 추천 정책을 학습하는 보수적 Q-러닝 모델

## 실행 방법

```bash
cd recsys-ex/offline-rl
python offline_rl.py
```

의존성: `torch`, `numpy`

출력 예시:
- 20 에폭마다 TD loss, CQL 페널티, 정책 정확도
- 데이터 정책 vs 학습된 정책 비교

## 핵심 개념

### 문제 설정

```
상태(State):  유저 특성 (8차원 벡터)
행동(Action): 추천 아이템 선택 (10개 중 1개)
보상(Reward): 클릭(+1) / 미클릭(0) / 이탈(-1)
```

온라인 RL과 달리 환경과 직접 상호작용하지 않고, 사전 수집된 5000개 트랜지션만 사용

### DQN (Deep Q-Network)

```python
class QNetwork(nn.Module):
    # state -> Q값 (각 행동에 대한 기대 누적 보상)
    Linear(8, 64) + ReLU
    Linear(64, 64) + ReLU
    Linear(64, 10)  # 10개 행동 각각의 Q값
```

TD (Temporal Difference) 손실:

```
TD loss = MSE(Q(s, a), r + gamma * max_a' Q_target(s', a'))
```

| 기호 | 의미 |
|------|------|
| `Q(s, a)` | 현재 Q-network의 예측 |
| `Q_target` | 안정성을 위한 별도 target network |
| `gamma = 0.99` | 미래 보상 할인율 |

### CQL (Conservative Q-Learning)

표준 DQN의 문제점 (오프라인에서):
- 데이터에 없는 행동에 대한 Q값을 과대추정
- 실제 배포 시 이 행동을 선택 → 예상치 못한 실패

CQL 해결책:

```python
def cql_loss(q_values, actions, alpha=1.0):
    # 모든 행동의 Q값 합 (과대추정된 값 억제)
    logsumexp = logsumexp(Q(s, :))
    # 데이터에 있는 행동의 Q값 (보존)
    data_q = Q(s, a_data)
    return alpha * (logsumexp - data_q)
```

직관:
- `logsumexp(Q)`: 모든 행동 중 최대값에 근접 (분포 외 행동 포함)
- `Q(s, a_data)`: 실제 데이터의 행동 Q값
- 두 차이를 최소화 → 분포 외 행동의 Q값에 페널티

전체 손실:

```
L = TD_loss + alpha * CQL_penalty
```

### Target Network

```python
# 학습 네트워크 (매 스텝 업데이트)
q_net = QNetwork(...)
# 타겟 네트워크 (10 에폭마다 동기화)
target_net = QNetwork(...)
```

타겟 네트워크는 학습 안정성을 위해 주기적으로만 업데이트

### 정책 평가

```python
optimal_actions = states[:, :n_actions].argmax(dim=1)
policy_acc = (learned_actions == optimal_actions).mean()
```

## 관련 모델과의 비교

| 방법 | 환경 상호작용 | 분포 외 과대추정 | 보수성 제어 | 복잡도 |
|------|-------------|-----------------|------------|--------|
| **CQL (Offline RL)** | 불필요 | 억제 | alpha 파라미터 | 중간 |
| **표준 DQN (Online)** | 필요 | 발생 | 없음 | 낮음 |
| **BCQ** | 불필요 | 억제 | 행동 제약 | 높음 |
| **IQL** | 불필요 | 억제 | 암묵적 | 중간 |
| **Behavior Cloning** | 불필요 | 없음 (모방) | N/A | 낮음 |

### CQL vs 표준 DQN
- 표준 DQN: 온라인 탐색으로 Q값 과대추정 교정
- 오프라인 DQN: 탐색 불가 → 분포 외 행동의 Q값이 수렴 없이 상승
- CQL: 페널티 항으로 분포 외 Q값을 인위적으로 낮춤

### CQL vs BCQ (Batch Constrained Q-learning)
- BCQ: 행동 정책의 분포를 명시적으로 모델링하여 그 근방에서만 Q 업데이트
- CQL: 소프트 패널티로 더 단순하게 구현, alpha로 보수성 조절 가능

### CQL vs Behavior Cloning
- Behavior Cloning: 데이터 행동을 그대로 모방 (순방향 지도학습)
- CQL: 보상을 기반으로 데이터보다 나은 정책 탐색 가능
- 데이터 품질이 낮을수록 CQL의 장점이 부각됨

### 추천 시스템에서의 역할
- A/B 테스트 없이 로그 데이터로 정책 학습
- 탐색 비용이 높은 환경 (유저 이탈 위험)에서 안전한 정책 학습
- 클릭 로그 -> 장기 체류 시간 최적화

### 참고 논문
- Kumar et al., "Conservative Q-Learning for Offline Reinforcement Learning" (NeurIPS 2020)
- Fujimoto et al., "Off-Policy Deep Reinforcement Learning without Exploration" (ICML 2019)
- Mnih et al., "Human-level control through deep reinforcement learning" (Nature 2015)
