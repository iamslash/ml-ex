"""
Audio Classifier with PyTorch.
합성 오디오 특성으로 음성 명령어를 분류하는 경량 모델.
GPU 불필요, 외부 데이터 다운로드 불필요.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# --- 합성 오디오 데이터 ---
# 3가지 "음성 명령어" 패턴을 주파수 특성으로 표현
n_per_class = 200
n_mels = 20       # mel spectrogram bins
n_frames = 30     # 시간 프레임

def make_audio_data():
    X, y = [], []
    for _ in range(n_per_class):
        # Class 0: "yes" - 저주파 + 고주파 이중 피크
        spec = np.zeros((n_mels, n_frames))
        spec[2:5, :] = np.random.rand(3, n_frames) * 2 + 1
        spec[15:18, 10:20] = np.random.rand(3, 10) * 2 + 1
        spec += np.random.randn(n_mels, n_frames) * 0.3
        X.append(spec); y.append(0)

        # Class 1: "no" - 중주파 단일 피크, 짧은 지속
        spec = np.zeros((n_mels, n_frames))
        spec[8:12, 5:15] = np.random.rand(4, 10) * 3 + 1
        spec += np.random.randn(n_mels, n_frames) * 0.3
        X.append(spec); y.append(1)

        # Class 2: "stop" - 넓은 대역, 긴 지속
        spec = np.zeros((n_mels, n_frames))
        spec[3:17, 2:28] = np.random.rand(14, 26) * 1.5 + 0.5
        spec += np.random.randn(n_mels, n_frames) * 0.3
        X.append(spec); y.append(2)

    X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(1)  # (N, 1, n_mels, n_frames)
    y = torch.tensor(y)
    perm = torch.randperm(len(y))
    return X[perm], y[perm]

X, y = make_audio_data()
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"samples: {len(X)}, classes: 3 (yes/no/stop)")
print(f"spectrogram shape: ({n_mels}, {n_frames})")
print(f"train: {split}, test: {len(X) - split}")

# --- 모델: CNN for Audio ---
class AudioCNN(nn.Module):
    """Mel spectrogram을 입력으로 받는 2D CNN 분류기"""
    def __init__(self, n_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # (16, 20, 30)
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),                                # (16, 10, 15)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),   # (32, 10, 15)
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                # (32, 5, 7)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # (64, 5, 7)
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),                   # (64, 1, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = AudioCNN()
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

# --- 학습 ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
batch_size = 64

for epoch in range(30):
    model.train()
    perm = torch.randperm(split)
    total_loss = 0
    n_batches = 0

    for i in range(0, split, batch_size):
        idx = perm[i:i+batch_size]
        pred = model(X_train[idx])
        loss = criterion(pred, y_train[idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test).argmax(1)
            test_acc = (test_pred == y_test).float().mean().item() * 100
        print(f"epoch {epoch+1:2d} | loss {total_loss/n_batches:.4f} | test acc {test_acc:.1f}%")

# --- 클래스별 정확도 ---
print("\n--- per-class accuracy ---")
model.eval()
with torch.no_grad():
    preds = model(X_test).argmax(1)
    for c, name in enumerate(["yes", "no", "stop"]):
        mask = y_test == c
        acc = (preds[mask] == c).float().mean().item() * 100
        print(f"  {name}: {acc:.1f}%")
