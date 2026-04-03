# Q-Learning

> 강화학습의 기본 알고리즘. 에이전트가 환경과 상호작용하며 최적 행동을 학습한다.

## 실행 방법

```bash
cd scratch-ex/q-learning
python q_learning.py
```

## 핵심 개념

### 강화학습 프레임워크

```
에이전트 -- action --> 환경
에이전트 <-- state, reward -- 환경
```

- **상태 (State)**: 에이전트의 현재 위치 (r, c)
- **행동 (Action)**: 상하좌우 이동
- **보상 (Reward)**: 목표 +10, 함정 -10, 이동 -0.1
- **정책 (Policy)**: 각 상태에서 어떤 행동을 할지 결정

### Q-Value

Q(s, a) = 상태 s에서 행동 a를 하고, 이후 최적으로 행동했을 때의 기대 보상

### Q-Learning 업데이트 규칙

```
Q(s,a) = Q(s,a) + lr * [r + gamma * max_a' Q(s',a') - Q(s,a)]
```

| 항 | 의미 |
|---|------|
| r | 즉시 보상 |
| gamma * max Q(s',a') | 미래 보상의 추정값 (할인) |
| r + gamma * max Q(s',a') | TD target (목표값) |
| TD target - Q(s,a) | TD error (오차) |

### Epsilon-Greedy 정책

```
확률 epsilon → 랜덤 행동 (탐험, exploration)
확률 1-epsilon → 최대 Q값 행동 (활용, exploitation)
```

- 초기: epsilon=1.0 (거의 랜덤)
- 점차 감소 → 학습된 정책에 의존
- exploration-exploitation tradeoff의 핵심

### 지도학습과의 비교

| | 지도학습 | 강화학습 |
|---|---|---|
| 데이터 | (입력, 정답) 쌍 | (상태, 행동, 보상) 시퀀스 |
| 피드백 | 즉시, 정확 | 지연, 희소 |
| 목표 | 손실 최소화 | 누적 보상 최대화 |

## 환경: GridWorld

```
S . . .     S = 시작 (0,0)
. X . .     G = 목표 (3,3)
. . X .     X = 함정
. . . G     . = 빈칸
```

에이전트는 함정을 피해 목표에 도달하는 최적 경로를 학습한다.
