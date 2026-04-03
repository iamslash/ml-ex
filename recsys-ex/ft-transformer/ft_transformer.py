"""
FT-Transformer with PyTorch.
테이블 데이터를 위한 Feature Tokenizer + Transformer.
"""

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

# --- 합성 테이블 데이터 ---
n_samples = 2000
n_num_features = 8     # 수치형 특성
n_cat_features = 4     # 범주형 특성
n_cat_values = [5, 10, 3, 8]  # 각 범주형의 고유값 수

X_num = torch.randn(n_samples, n_num_features)
X_cat = [torch.randint(0, n, (n_samples,)) for n in n_cat_values]

# 정답: 수치형 + 범주형 조합
y = (X_num[:, 0] * X_num[:, 1] + X_num[:, 2] > 0).float()

split = int(0.8 * n_samples)
print(f"samples: {n_samples}, num features: {n_num_features}, cat features: {n_cat_features}")

# --- Feature Tokenizer ---
class FeatureTokenizer(nn.Module):
    """각 특성을 d_token 차원 토큰으로 변환"""
    def __init__(self, n_num, n_cat_values, d_token):
        super().__init__()
        # 수치형: 각 특성마다 linear projection
        self.num_tokenizers = nn.ModuleList([
            nn.Linear(1, d_token) for _ in range(n_num)
        ])
        # 범주형: 각 특성마다 embedding
        self.cat_tokenizers = nn.ModuleList([
            nn.Embedding(n_vals, d_token) for n_vals in n_cat_values
        ])
        # [CLS] 토큰
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)

    def forward(self, x_num, x_cat_list):
        tokens = []
        # 수치형 토큰
        for i, tokenizer in enumerate(self.num_tokenizers):
            tokens.append(tokenizer(x_num[:, i:i+1]))  # (batch, d_token)
        # 범주형 토큰
        for i, tokenizer in enumerate(self.cat_tokenizers):
            tokens.append(tokenizer(x_cat_list[i]))  # (batch, d_token)

        tokens = torch.stack(tokens, dim=1)  # (batch, n_features, d_token)
        # [CLS] 토큰 추가
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        return torch.cat([cls, tokens], dim=1)  # (batch, 1+n_features, d_token)

# --- FT-Transformer ---
class FTTransformer(nn.Module):
    def __init__(self, n_num, n_cat_values, d_token=32, n_heads=4, n_layers=2, d_ff=64):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_num, n_cat_values, d_token)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_ff,
            batch_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(d_token, 1)

    def forward(self, x_num, x_cat_list):
        tokens = self.tokenizer(x_num, x_cat_list)  # (batch, 1+n_features, d_token)
        out = self.transformer(tokens)                # (batch, 1+n_features, d_token)
        cls_out = out[:, 0, :]                        # (batch, d_token) [CLS] token
        return self.fc_out(cls_out).squeeze()

model = FTTransformer(n_num_features, n_cat_values, d_token=32, n_heads=4, n_layers=2)
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

# --- 학습 ---
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
batch_size = 128

for epoch in range(50):
    model.train()
    perm = torch.randperm(split)
    total_loss = 0
    n_batches = 0

    for i in range(0, split, batch_size):
        idx = perm[i:i+batch_size]
        pred = model(X_num[idx], [c[idx] for c in X_cat])
        loss = criterion(pred, y[idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_pred = model(X_num[split:], [c[split:] for c in X_cat])
            test_acc = ((test_pred > 0).float() == y[split:]).float().mean().item() * 100
        print(f"epoch {epoch+1:2d} | loss {total_loss/n_batches:.4f} | test acc {test_acc:.1f}%")
