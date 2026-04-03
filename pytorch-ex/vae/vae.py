"""
VAE (Variational Autoencoder) with PyTorch.
MNIST 이미지 생성.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# --- 데이터 ---
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST('.data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# --- 모델 ---
class VAE(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.enc_fc1 = nn.Linear(784, 256)
        self.enc_mu = nn.Linear(256, latent_dim)      # 평균
        self.enc_logvar = nn.Linear(256, latent_dim)   # 로그 분산

        # Decoder
        self.dec_fc1 = nn.Linear(latent_dim, 256)
        self.dec_fc2 = nn.Linear(256, 784)

    def encode(self, x):
        h = F.relu(self.enc_fc1(x))
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + std * eps"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        h = F.relu(self.dec_fc1(z))
        return torch.sigmoid(self.dec_fc2(h))

    def forward(self, x):
        x_flat = x.view(-1, 784)
        mu, logvar = self.encode(x_flat)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

def vae_loss(x_recon, x, mu, logvar):
    """Reconstruction loss + KL divergence"""
    x_flat = x.view(-1, 784)
    recon_loss = F.binary_cross_entropy(x_recon, x_flat, reduction='sum')
    # KL divergence: D_KL(q(z|x) || p(z)) where p(z) = N(0,1)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

latent_dim = 8
model = VAE(latent_dim).to(device)
print(f"latent_dim: {latent_dim}")
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

# --- 학습 ---
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for data, _ in train_loader:
        data = data.to(device)

        optimizer.zero_grad()
        x_recon, mu, logvar = model(data)
        loss = vae_loss(x_recon, data, mu, logvar)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataset)

    if (epoch + 1) % 2 == 0:
        print(f"epoch {epoch+1:2d}/{num_epochs} | loss {avg_loss:.2f}")

# --- 새로운 이미지 생성 ---
model.eval()
with torch.no_grad():
    # 잠재 공간에서 랜덤 샘플링
    z = torch.randn(10, latent_dim).to(device)
    generated = model.decode(z)  # (10, 784)

    print(f"\ngenerated {len(z)} images from random latent vectors")
    for i in range(3):
        pixels = generated[i].cpu().numpy()
        active = (pixels > 0.5).sum()
        print(f"  sample {i}: {active} active pixels (of 784)")

    # 기존 이미지의 잠재 표현
    sample, label = train_dataset[0]
    _, mu, logvar = model(sample.unsqueeze(0).to(device))
    print(f"\ndigit {label} latent mu: {mu[0].cpu().numpy().round(2)}")
