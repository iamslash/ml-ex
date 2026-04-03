"""
Autoencoder with PyTorch.
MNIST 이미지를 압축/복원하는 오토인코더.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# --- 데이터 ---
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST('.data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('.data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)

print(f"train: {len(train_dataset)}, test: {len(test_dataset)}")

# --- 모델 ---
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        # Encoder: 784 -> 128 -> latent_dim
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        # Decoder: latent_dim -> 128 -> 784
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),  # 출력을 0~1로
        )

    def forward(self, x):
        z = self.encoder(x)           # (batch, latent_dim)
        x_recon = self.decoder(z)     # (batch, 784)
        return x_recon, z

latent_dim = 16
model = Autoencoder(latent_dim).to(device)
print(f"\nlatent_dim: {latent_dim}")
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

# --- 학습 ---
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for data, _ in train_loader:  # 라벨 사용하지 않음
        data = data.to(device)
        x_flat = data.view(data.size(0), -1)  # (batch, 784)

        optimizer.zero_grad()
        x_recon, z = model(data)
        loss = criterion(x_recon, x_flat)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    if (epoch + 1) % 2 == 0:
        print(f"epoch {epoch+1:2d}/{num_epochs} | loss {avg_loss:.6f}")

# --- 평가 ---
model.eval()
test_loss = 0
with torch.no_grad():
    for data, _ in test_loader:
        data = data.to(device)
        x_flat = data.view(data.size(0), -1)
        x_recon, z = model(data)
        test_loss += criterion(x_recon, x_flat).item()

print(f"\ntest loss: {test_loss / len(test_loader):.6f}")

# 압축률 확인
original_size = 28 * 28
print(f"compression: {original_size} -> {latent_dim} ({original_size/latent_dim:.0f}x)")

# 잠재 공간 분석
with torch.no_grad():
    sample, label = test_dataset[0]
    _, z = model(sample.unsqueeze(0).to(device))
    print(f"\nsample digit: {label}")
    print(f"latent vector: {z[0].cpu().numpy().round(2)}")
