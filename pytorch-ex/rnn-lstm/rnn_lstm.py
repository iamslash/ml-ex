"""
RNN/LSTM with PyTorch.
시퀀스 패턴 분류: 3가지 패턴의 토큰 시퀀스를 분류.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# --- 합성 데이터 ---
# 3가지 시퀀스 패턴: 상승(0), 하강(1), 진동(2)
vocab_size = 30
seq_len = 20
n_per_class = 1000

def make_data(n_per_class):
    X, y = [], []
    for _ in range(n_per_class):
        # Class 0: 상승 패턴 (작은 값 → 큰 값)
        seq = torch.sort(torch.randint(0, vocab_size, (seq_len,)))[0]
        seq += torch.randint(0, 3, (seq_len,))  # noise
        X.append(seq.clamp(0, vocab_size - 1)); y.append(0)

        # Class 1: 하강 패턴 (큰 값 → 작은 값)
        seq = torch.sort(torch.randint(0, vocab_size, (seq_len,)), descending=True)[0]
        seq += torch.randint(0, 3, (seq_len,))
        X.append(seq.clamp(0, vocab_size - 1)); y.append(1)

        # Class 2: 진동 패턴 (high-low 반복)
        seq = torch.zeros(seq_len, dtype=torch.long)
        for i in range(seq_len):
            if i % 2 == 0:
                seq[i] = torch.randint(20, vocab_size, (1,))
            else:
                seq[i] = torch.randint(0, 10, (1,))
        X.append(seq); y.append(2)

    X = torch.stack(X)
    y = torch.tensor(y)
    perm = torch.randperm(len(y))
    return X[perm], y[perm]

X_all, y_all = make_data(n_per_class)
split = int(0.8 * len(X_all))
X_train, X_test = X_all[:split], X_all[split:]
y_train, y_test = y_all[:split], y_all[split:]

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256)

print(f"train: {len(X_train)}, test: {len(X_test)}")
print(f"vocab: {vocab_size}, seq_len: {seq_len}, classes: 3")

# --- 모델 ---
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        x = self.embedding(x)
        output, (h_n, c_n) = self.lstm(x)
        h_last = h_n.squeeze(0)
        return self.fc(h_last)

model = LSTMClassifier(vocab_size, embed_dim=16, hidden_dim=32, n_classes=3).to(device)
print(f"\n{model}")
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

# --- 학습 ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 15

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (output.argmax(1) == y_batch).sum().item()
        total += y_batch.size(0)

    train_acc = correct / total * 100

    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            test_correct += (output.argmax(1) == y_batch).sum().item()
            test_total += y_batch.size(0)

    test_acc = test_correct / test_total * 100
    avg_loss = total_loss / len(train_loader)
    print(f"epoch {epoch+1:2d}/{num_epochs} | loss {avg_loss:.4f} | train {train_acc:.1f}% | test {test_acc:.1f}%")

# --- 클래스별 정확도 ---
print("\n--- per-class accuracy ---")
model.eval()
with torch.no_grad():
    preds = model(X_test.to(device)).argmax(1).cpu()
    for c, name in enumerate(["ascending", "descending", "oscillating"]):
        mask = y_test == c
        acc = (preds[mask] == c).float().mean().item() * 100
        print(f"  {name}: {acc:.1f}%")
