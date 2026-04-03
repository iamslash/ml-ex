# DQN (Deep Q-Network)

> 신경망으로 Q-value를 근사하는 강화학습. Q-Learning의 딥러닝 확장.

## 실행 방법

```bash
cd pytorch-ex/dqn
python dqn.py
```

## 핵심 개념

### Q-Learning → DQN

| | Q-Learning (scratch) | DQN (이 코드) |
|---|---|---|
| Q 저장 | 테이블 (유한 상태) | 신경망 (연속 상태) |
| 상태 공간 | 이산 (그리드 위치) | 연속 (위치, 속도, 각도) |
| 일반화 | 불가 | 유사 상태에 일반화 |

### DQN의 핵심 기법

#### 1. Experience Replay

```
(state, action, reward, next_state, done) → 버퍼에 저장
학습 시 버퍼에서 랜덤 배치 샘플링
```

- 데이터 재사용으로 효율적 학습
- 시간적 상관관계 제거

#### 2. Target Network

```
Q_target = r + gamma * max_a' Q_target(s', a')
```

- target 계산에 별도의 네트워크 사용
- 주기적으로 policy_net → target_net 복사
- 학습 안정성 확보

### 학습 루프

```
1. epsilon-greedy로 행동 선택
2. 환경에서 (next_state, reward, done) 관측
3. 경험을 Replay Buffer에 저장
4. 버퍼에서 배치 샘플링
5. Q_pred = policy_net(state)[action]
6. Q_target = reward + gamma * max(target_net(next_state))
7. Loss = MSE(Q_pred, Q_target)
8. 역전파 + 파라미터 갱신
```

## 환경: CartPole

막대가 달린 카트를 좌우로 밀어서 막대가 쓰러지지 않도록 균형을 유지한다.
- 상태: [위치, 속도, 각도, 각속도] (연속 4차원)
- 행동: 왼쪽(0), 오른쪽(1)
- 보상: 막대가 서있으면 +1, 쓰러지면 0
