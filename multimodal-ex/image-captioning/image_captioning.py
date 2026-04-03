"""
Image Captioning with PyTorch.
이미지 특성 → 텍스트 시퀀스 생성 (encoder-decoder).
"""

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

# --- 합성 데이터 ---
# 이미지: 특성 벡터, 캡션: 토큰 시퀀스
img_dim = 32
vocab_size = 30
max_caption_len = 8
PAD, BOS, EOS = 0, 1, 2

n_pairs = 2000

# 이미지 특성
image_features = torch.randn(n_pairs, img_dim)

# 캡션: 이미지 특성에 기반한 패턴
captions = []
for i in range(n_pairs):
    # 이미지 특성의 상위 3개 인덱스를 캡션 토큰으로 사용
    top3 = image_features[i, :vocab_size - 3].abs().topk(3).indices
    tokens = [BOS] + [t.item() + 3 for t in top3]  # offset by 3 (PAD, BOS, EOS)
    tokens += [EOS]
    tokens += [PAD] * (max_caption_len - len(tokens))
    captions.append(tokens[:max_caption_len])

captions = torch.tensor(captions)
split = int(0.8 * n_pairs)

print(f"pairs: {n_pairs}, img_dim: {img_dim}, vocab: {vocab_size}")
print(f"train: {split}, test: {n_pairs - split}")

# --- 모델 ---
class ImageCaptioner(nn.Module):
    def __init__(self, img_dim, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        # 이미지 인코더
        self.img_proj = nn.Linear(img_dim, hidden_dim)

        # 캡션 디코더 (LSTM)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, images, captions):
        # 이미지 → 초기 은닉 상태
        img_feat = self.img_proj(images)  # (batch, hidden)
        h0 = img_feat.unsqueeze(0)        # (1, batch, hidden)
        c0 = torch.zeros_like(h0)

        # 캡션 디코딩 (teacher forcing)
        cap_emb = self.embedding(captions)  # (batch, seq_len, embed)
        output, _ = self.lstm(cap_emb, (h0, c0))  # (batch, seq_len, hidden)
        logits = self.fc_out(output)  # (batch, seq_len, vocab)
        return logits

    def generate(self, images, max_len=8):
        batch = images.size(0)
        img_feat = self.img_proj(images)
        h = img_feat.unsqueeze(0)
        c = torch.zeros_like(h)

        token = torch.full((batch, 1), BOS, dtype=torch.long)
        generated = [token]

        for _ in range(max_len - 1):
            emb = self.embedding(token)
            output, (h, c) = self.lstm(emb, (h, c))
            logits = self.fc_out(output)
            token = logits.argmax(dim=-1)
            generated.append(token)

        return torch.cat(generated, dim=1)

model = ImageCaptioner(img_dim, vocab_size)
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

# --- 학습 ---
criterion = nn.CrossEntropyLoss(ignore_index=PAD)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(200):
    model.train()
    # Teacher forcing: 입력 = captions[:, :-1], 정답 = captions[:, 1:]
    imgs = image_features[:split]
    caps_in = captions[:split, :-1]
    caps_target = captions[:split, 1:]

    logits = model(imgs, caps_in)
    loss = criterion(logits.reshape(-1, vocab_size), caps_target.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 40 == 0:
        model.eval()
        with torch.no_grad():
            test_logits = model(image_features[split:], captions[split:, :-1])
            test_loss = criterion(test_logits.reshape(-1, vocab_size), captions[split:, 1:].reshape(-1))
            # 토큰 정확도
            preds = test_logits.argmax(-1)
            mask = captions[split:, 1:] != PAD
            acc = (preds[mask] == captions[split:, 1:][mask]).float().mean().item() * 100
        print(f"epoch {epoch+1:3d} | loss {loss.item():.4f} | test loss {test_loss.item():.4f} | token acc {acc:.1f}%")

# --- 생성 ---
print("\n--- caption generation ---")
model.eval()
with torch.no_grad():
    test_imgs = image_features[split:split+5]
    generated = model.generate(test_imgs)

    for i in range(5):
        expected = [t.item() for t in captions[split+i] if t.item() not in (PAD, BOS, EOS)]
        pred = [t.item() for t in generated[i] if t.item() not in (PAD, BOS, EOS)]
        match = expected == pred
        print(f"  expected: {expected} | generated: {pred} [{'OK' if match else 'MISS'}]")
