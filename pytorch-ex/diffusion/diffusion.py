"""
Toy Diffusion Model with PyTorch.
2D 점 데이터에 대한 DDPM (Denoising Diffusion Probabilistic Model).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# --- 2D 데이터 생성 (나선형) ---
def make_spiral(n=1000):
    t = torch.linspace(0, 4 * np.pi, n)
    x = t * torch.cos(t) / (4 * np.pi)
    y = t * torch.sin(t) / (4 * np.pi)
    data = torch.stack([x, y], dim=1)  # (n, 2)
    data += torch.randn_like(data) * 0.02  # noise
    return data

data = make_spiral(2000).to(device)
print(f"data shape: {data.shape}")

# --- Noise schedule ---
T = 100  # diffusion steps
beta = torch.linspace(1e-4, 0.02, T).to(device)
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

print(f"T={T}, beta range: [{beta[0]:.4f}, {beta[-1]:.4f}]")

# --- Denoising network ---
class DenoiseNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 입력: x_t (2) + t_emb (16) = 18
        self.time_emb = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(18, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # 예측된 노이즈
        )

    def forward(self, x_t, t):
        # t: (batch,) -> (batch, 16)
        t_emb = self.time_emb(t.float().unsqueeze(1) / T)
        x_input = torch.cat([x_t, t_emb], dim=1)  # (batch, 18)
        return self.net(x_input)

model = DenoiseNet().to(device)
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

# --- Forward diffusion: q(x_t | x_0) ---
def forward_diffusion(x_0, t):
    """x_0에 t 스텝만큼 노이즈를 추가"""
    noise = torch.randn_like(x_0)
    ab = alpha_bar[t].unsqueeze(1)  # (batch, 1)
    x_t = torch.sqrt(ab) * x_0 + torch.sqrt(1 - ab) * noise
    return x_t, noise

# --- 학습 ---
optimizer = optim.Adam(model.parameters(), lr=0.001)
batch_size = 256
num_epochs = 200

for epoch in range(num_epochs):
    # 랜덤 배치 선택
    idx = torch.randint(0, len(data), (batch_size,))
    x_0 = data[idx]

    # 랜덤 타임스텝
    t = torch.randint(0, T, (batch_size,)).to(device)

    # Forward diffusion
    x_t, noise = forward_diffusion(x_0, t)

    # 노이즈 예측
    noise_pred = model(x_t, t)

    # Loss: 실제 노이즈와 예측 노이즈의 MSE
    loss = nn.MSELoss()(noise_pred, noise)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"epoch {epoch+1:3d}/{num_epochs} | loss {loss.item():.6f}")

# --- Reverse diffusion: 샘플 생성 ---
@torch.no_grad()
def sample(n_samples=500):
    model.eval()
    x = torch.randn(n_samples, 2).to(device)  # 순수 노이즈에서 시작

    for t in reversed(range(T)):
        t_batch = torch.full((n_samples,), t, device=device)
        noise_pred = model(x, t_batch)

        b = beta[t]
        a = alpha[t]
        ab = alpha_bar[t]

        # DDPM 역방향 스텝
        x = (1 / torch.sqrt(a)) * (x - (b / torch.sqrt(1 - ab)) * noise_pred)

        if t > 0:
            x += torch.sqrt(b) * torch.randn_like(x)

    return x.cpu().numpy()

generated = sample(500)
original = data.cpu().numpy()

print(f"\n--- results ---")
print(f"original data  - mean: ({original.mean(0)[0]:.3f}, {original.mean(0)[1]:.3f}), "
      f"std: ({original.std(0)[0]:.3f}, {original.std(0)[1]:.3f})")
print(f"generated data - mean: ({generated.mean(0)[0]:.3f}, {generated.mean(0)[1]:.3f}), "
      f"std: ({generated.std(0)[0]:.3f}, {generated.std(0)[1]:.3f})")
