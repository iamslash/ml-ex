"""
LSTM-FCN (LSTM + Fully Convolutional Network) with PyTorch.
시계열 분류를 위한 병렬 LSTM + CNN 아키텍처.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# --- 합성 시계열 분류 데이터 ---
n_per_class = 200
seq_len = 50

def make_data():
    X, y = [], []
    for _ in range(n_per_class):
        # Class 0: sin 패턴
        t = np.linspace(0, 4 * np.pi, seq_len)
        X.append(np.sin(t) + 0.3 * np.random.randn(seq_len))
        y.append(0)

        # Class 1: sawtooth 패턴
        t = np.linspace(0, 4, seq_len)
        X.append((t % 1) + 0.3 * np.random.randn(seq_len))
        y.append(1)

        # Class 2: step 패턴
        step = np.zeros(seq_len)
        step[seq_len//4:seq_len//2] = 1
        step[3*seq_len//4:] = -1
        X.append(step + 0.3 * np.random.randn(seq_len))
        y.append(2)

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(y)
    perm = torch.randperm(len(y))
    return X[perm], y[perm]

X, y = make_data()
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"samples: {len(X)}, seq_len: {seq_len}, classes: 3")
print(f"train: {split}, test: {len(X) - split}")

# --- LSTM-FCN 모델 ---
class LSTMFCN(nn.Module):
    """
    LSTM-FCN: LSTM 경로와 CNN 경로를 병렬로 실행 후 결합.
    LSTM: 시간적 의존성 포착
    FCN: 로컬 패턴 포착 + GlobalAveragePooling
    """
    def __init__(self, input_dim=1, n_classes=3, lstm_hidden=64, n_layers=1):
        super().__init__()
        # LSTM 경로
        self.lstm = nn.LSTM(input_dim, lstm_hidden, n_layers, batch_first=True)
        self.lstm_dropout = nn.Dropout(0.2)

        # FCN 경로 (1D CNN)
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=8, padding=4),
            nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
        )
        # Global Average Pooling은 forward에서 수행

        # 분류 헤드: LSTM 마지막 은닉 + GAP 결합
        self.fc = nn.Linear(lstm_hidden + 128, n_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        x_lstm = x.unsqueeze(-1)  # (batch, seq_len, 1)
        x_cnn = x.unsqueeze(1)    # (batch, 1, seq_len)

        # LSTM 경로
        lstm_out, (h_n, _) = self.lstm(x_lstm)
        lstm_feat = self.lstm_dropout(h_n[-1])  # (batch, lstm_hidden)

        # FCN 경로
        c = self.conv1(x_cnn)   # (batch, 128, seq_len)
        c = self.conv2(c)       # (batch, 256, seq_len)
        c = self.conv3(c)       # (batch, 128, seq_len)
        cnn_feat = c.mean(dim=2)  # Global Average Pooling → (batch, 128)

        # 결합
        combined = torch.cat([lstm_feat, cnn_feat], dim=1)
        return self.fc(combined)

model = LSTMFCN()
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

# --- 학습 ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
batch_size = 64

for epoch in range(50):
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

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            test_acc = (test_pred.argmax(1) == y_test).float().mean().item() * 100
        print(f"epoch {epoch+1:2d} | loss {total_loss/n_batches:.4f} | test acc {test_acc:.1f}%")

# --- 클래스별 정확도 ---
print("\n--- per-class accuracy ---")
model.eval()
with torch.no_grad():
    preds = model(X_test).argmax(1)
    for c, name in enumerate(["sin", "sawtooth", "step"]):
        mask = y_test == c
        acc = (preds[mask] == c).float().mean().item() * 100
        print(f"  {name}: {acc:.1f}%")
