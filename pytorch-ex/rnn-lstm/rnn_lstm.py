"""
RNN/LSTM with PyTorch.
IMDB 유사 감성 분류 (합성 데이터).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# --- 합성 데이터 ---
# 시퀀스 내 특정 패턴으로 긍정/부정 분류
vocab_size = 50
seq_len = 20
n_samples = 2000

def make_data(n):
    X = torch.randint(0, vocab_size, (n, seq_len))
    # 규칙: 시퀀스 앞쪽에 높은 값(25-49)이 많으면 긍정(1)
    first_half_mean = X[:, :seq_len//2].float().mean(dim=1)
    y = (first_half_mean > vocab_size / 2).long()
    return X, y

X_train, y_train = make_data(n_samples)
X_test, y_test = make_data(500)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256)

print(f"train: {len(X_train)}, test: {len(X_test)}")
print(f"vocab: {vocab_size}, seq_len: {seq_len}")

# --- 모델 ---
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)            # (batch, seq_len, embed_dim)
        output, (h_n, c_n) = self.lstm(x)  # h_n: (1, batch, hidden_dim)
        h_last = h_n.squeeze(0)           # (batch, hidden_dim)
        logits = self.fc(h_last)          # (batch, n_classes)
        return logits

model = LSTMClassifier(vocab_size, embed_dim=16, hidden_dim=32, n_classes=2).to(device)
print(f"\n{model}")
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

# --- 학습 ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

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

    # 테스트
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

print(f"\nfinal test accuracy: {test_acc:.1f}%")
