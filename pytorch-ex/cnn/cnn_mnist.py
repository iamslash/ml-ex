"""
CNN with PyTorch.
MNIST 손글씨 숫자 분류.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- 설정 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# --- 데이터 ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('.data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('.data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

print(f"train: {len(train_dataset)}, test: {len(test_dataset)}")
print(f"image shape: {train_dataset[0][0].shape}")  # (1, 28, 28)

# --- 모델 ---
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # (16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2),                                # (16, 14, 14)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),   # (32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),                                # (32, 7, 7)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),                                   # (32*7*7,)
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

model = CNN().to(device)
print(f"\n{model}")
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

# --- 학습 설정 ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 5

# --- 학습 ---
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    train_acc = correct / total * 100
    avg_loss = total_loss / len(train_loader)

    # 테스트
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            test_correct += pred.eq(target).sum().item()
            test_total += target.size(0)

    test_acc = test_correct / test_total * 100
    print(f"epoch {epoch+1}/{num_epochs} | loss {avg_loss:.4f} | train acc {train_acc:.1f}% | test acc {test_acc:.1f}%")

print(f"\nfinal test accuracy: {test_acc:.1f}%")
