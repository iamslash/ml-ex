"""
Q-Learning from scratch.
순수 numpy로 구현한 강화학습. 간단한 GridWorld 환경.
"""

import numpy as np

np.random.seed(42)

# --- GridWorld 환경 ---
# 4x4 그리드. 에이전트는 (0,0)에서 시작, (3,3)이 목표.
# 함정(trap)에 빠지면 -10 보상, 목표 도달 시 +10 보상, 이동 시 -0.1.
#
#  S . . .
#  . X . .
#  . . X .
#  . . . G
#
# S=시작, G=목표, X=함정, .=빈칸

GRID_SIZE = 4
START = (0, 0)
GOAL = (3, 3)
TRAPS = [(1, 1), (2, 2)]

ACTIONS = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}
ACTION_NAMES = {0: "up", 1: "down", 2: "left", 3: "right"}
n_actions = len(ACTIONS)

def step(state, action):
    """환경에서 한 스텝 진행. (next_state, reward, done) 반환."""
    dr, dc = ACTIONS[action]
    r, c = state
    nr, nc = r + dr, c + dc

    # 벽에 부딪히면 제자리
    if nr < 0 or nr >= GRID_SIZE or nc < 0 or nc >= GRID_SIZE:
        nr, nc = r, c

    next_state = (nr, nc)

    if next_state == GOAL:
        return next_state, 10.0, True
    elif next_state in TRAPS:
        return next_state, -10.0, True
    else:
        return next_state, -0.1, False

# --- Q-Table 초기화 ---
Q = np.zeros((GRID_SIZE, GRID_SIZE, n_actions))

# --- 하이퍼파라미터 ---
learning_rate = 0.1
gamma = 0.95          # 할인율
epsilon = 1.0         # 탐험율
epsilon_decay = 0.995
epsilon_min = 0.01
num_episodes = 1000

# --- 학습 ---
rewards_history = []

for episode in range(num_episodes):
    state = START
    total_reward = 0
    done = False
    steps = 0

    while not done and steps < 50:
        r, c = state

        # epsilon-greedy 정책
        if np.random.rand() < epsilon:
            action = np.random.randint(n_actions)  # 탐험
        else:
            action = np.argmax(Q[r, c])  # 활용

        next_state, reward, done = step(state, action)
        nr, nc = next_state

        # Q-Learning 업데이트
        # Q(s,a) = Q(s,a) + lr * [r + gamma * max_a' Q(s',a') - Q(s,a)]
        best_next = np.max(Q[nr, nc])
        td_target = reward + gamma * best_next * (1 - done)
        td_error = td_target - Q[r, c, action]
        Q[r, c, action] += learning_rate * td_error

        state = next_state
        total_reward += reward
        steps += 1

    rewards_history.append(total_reward)

    # epsilon 감소
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(rewards_history[-100:])
        print(f"episode {episode+1:4d} | avg reward {avg_reward:7.2f} | epsilon {epsilon:.3f}")

# --- 학습된 정책 출력 ---
print("\n--- learned policy ---")
symbols = {"up": "^", "down": "v", "left": "<", "right": ">"}

for r in range(GRID_SIZE):
    row = ""
    for c in range(GRID_SIZE):
        if (r, c) == GOAL:
            row += " G "
        elif (r, c) in TRAPS:
            row += " X "
        else:
            best_action = np.argmax(Q[r, c])
            row += f" {symbols[ACTION_NAMES[best_action]]} "
    print(row)

# --- Q-Table 값 ---
print("\n--- Q-values at key positions ---")
for pos, name in [(START, "start(0,0)"), ((0, 3), "(0,3)"), ((3, 0), "(3,0)")]:
    r, c = pos
    vals = ", ".join(f"{ACTION_NAMES[a]}={Q[r,c,a]:.2f}" for a in range(n_actions))
    print(f"  {name}: {vals}")

# --- 학습된 에이전트 실행 ---
print("\n--- agent trajectory ---")
state = START
trajectory = [state]

for _ in range(20):
    r, c = state
    action = np.argmax(Q[r, c])
    next_state, reward, done = step(state, action)
    trajectory.append(next_state)
    state = next_state
    if done:
        break

print(" -> ".join(str(s) for s in trajectory))
result = "GOAL!" if trajectory[-1] == GOAL else "TRAP!" if trajectory[-1] in TRAPS else "timeout"
print(f"result: {result}")
