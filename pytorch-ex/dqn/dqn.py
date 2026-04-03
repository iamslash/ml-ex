"""
DQN (Deep Q-Network) with PyTorch.
CartPole 환경에서 신경망으로 Q-value를 근사.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# --- 간단한 CartPole 유사 환경 ---
# 실제 gym 없이 핵심 개념 학습을 위한 간단한 환경
class SimpleCartPole:
    """
    상태: [위치, 속도, 각도, 각속도]
    행동: 0 (왼쪽), 1 (오른쪽)
    목표: 막대가 쓰러지지 않도록 균형 유지
    """
    def __init__(self):
        self.gravity = 9.8
        self.cart_mass = 1.0
        self.pole_mass = 0.1
        self.pole_length = 0.5
        self.force_mag = 10.0
        self.dt = 0.02
        self.state = None

    def reset(self):
        self.state = np.random.uniform(-0.05, 0.05, size=4)
        return self.state.copy()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag

        total_mass = self.cart_mass + self.pole_mass
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # 물리 시뮬레이션
        temp = (force + self.pole_mass * self.pole_length * theta_dot**2 * sin_theta) / total_mass
        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / \
                    (self.pole_length * (4/3 - self.pole_mass * cos_theta**2 / total_mass))
        x_acc = temp - self.pole_mass * self.pole_length * theta_acc * cos_theta / total_mass

        x += self.dt * x_dot
        x_dot += self.dt * x_acc
        theta += self.dt * theta_dot
        theta_dot += self.dt * theta_acc

        self.state = np.array([x, x_dot, theta, theta_dot])

        # 종료 조건
        done = abs(x) > 2.4 or abs(theta) > 0.21  # ~12 degrees
        reward = 1.0 if not done else 0.0

        return self.state.copy(), reward, done

# --- DQN 모델 ---
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.net(x)

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)

# --- 설정 ---
env = SimpleCartPole()
state_dim = 4
action_dim = 2

policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
buffer = ReplayBuffer()

print(f"params: {sum(p.numel() for p in policy_net.parameters()):,}")

# 하이퍼파라미터
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 64
target_update = 10
num_episodes = 300

# --- 학습 ---
rewards_history = []

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    for step in range(200):
        # Epsilon-greedy
        if random.random() < epsilon:
            action = random.randint(0, action_dim - 1)
        else:
            with torch.no_grad():
                q_values = policy_net(torch.FloatTensor(state))
                action = q_values.argmax().item()

        next_state, reward, done = env.step(action)
        buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # 학습
        if len(buffer) >= batch_size:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)

            states_t = torch.FloatTensor(states)
            actions_t = torch.LongTensor(actions)
            rewards_t = torch.FloatTensor(rewards)
            next_states_t = torch.FloatTensor(next_states)
            dones_t = torch.FloatTensor(dones)

            # 현재 Q값
            q_values = policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()

            # Target Q값 (target network 사용)
            with torch.no_grad():
                next_q = target_net(next_states_t).max(1)[0]
                target = rewards_t + gamma * next_q * (1 - dones_t)

            loss = nn.MSELoss()(q_values, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    rewards_history.append(total_reward)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Target network 업데이트
    if (episode + 1) % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if (episode + 1) % 30 == 0:
        avg = np.mean(rewards_history[-30:])
        print(f"episode {episode+1:3d} | avg reward {avg:6.1f} | epsilon {epsilon:.3f}")

# --- 최종 평가 ---
eval_rewards = []
for _ in range(20):
    state = env.reset()
    total_reward = 0
    for _ in range(200):
        with torch.no_grad():
            action = policy_net(torch.FloatTensor(state)).argmax().item()
        state, reward, done = env.step(action)
        total_reward += reward
        if done:
            break
    eval_rewards.append(total_reward)

print(f"\neval (20 episodes): mean {np.mean(eval_rewards):.1f}, max {np.max(eval_rewards):.0f}")
