"""
CLIP-style Contrastive Learning with PyTorch.
이미지-텍스트 매칭을 위한 contrastive 학습.
"""

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

# --- 합성 데이터 ---
# 이미지/텍스트를 특성 벡터로 표현
n_pairs = 200
img_dim = 32
txt_dim = 24
embed_dim = 16

# 이미지-텍스트 쌍 (같은 인덱스끼리 매칭)
image_features = torch.randn(n_pairs, img_dim)
text_features = torch.randn(n_pairs, txt_dim)

# 매칭되는 쌍은 비슷한 잠재 요인 공유
shared_factor = torch.randn(n_pairs, 8)
image_features[:, :8] += shared_factor * 2
text_features[:, :8] += shared_factor * 2

split = int(0.8 * n_pairs)
print(f"pairs: {n_pairs}, train: {split}, test: {n_pairs - split}")

# --- CLIP 모델 ---
class ImageEncoder(nn.Module):
    def __init__(self, in_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, embed_dim),
        )

    def forward(self, x):
        emb = self.net(x)
        return emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)

class TextEncoder(nn.Module):
    def __init__(self, in_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, embed_dim),
        )

    def forward(self, x):
        emb = self.net(x)
        return emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)

class CLIPModel(nn.Module):
    def __init__(self, img_dim, txt_dim, embed_dim):
        super().__init__()
        self.image_encoder = ImageEncoder(img_dim, embed_dim)
        self.text_encoder = TextEncoder(txt_dim, embed_dim)
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, images, texts):
        img_emb = self.image_encoder(images)
        txt_emb = self.text_encoder(texts)
        return img_emb, txt_emb

model = CLIPModel(img_dim, txt_dim, embed_dim)
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

# --- Contrastive Loss (InfoNCE) ---
def clip_loss(img_emb, txt_emb, temperature):
    logits = img_emb @ txt_emb.T / temperature.exp()
    labels = torch.arange(len(img_emb))
    loss_i2t = nn.CrossEntropyLoss()(logits, labels)
    loss_t2i = nn.CrossEntropyLoss()(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2

# --- 학습 ---
optimizer = optim.Adam(model.parameters(), lr=0.001)
batch_size = 64

for epoch in range(100):
    model.train()
    perm = torch.randperm(split)
    total_loss = 0
    n_batches = 0

    for i in range(0, split, batch_size):
        idx = perm[i:i+batch_size]
        img_emb, txt_emb = model(image_features[idx], text_features[idx])
        loss = clip_loss(img_emb, txt_emb, model.temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if (epoch + 1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            test_img, test_txt = model(image_features[split:], text_features[split:])
            scores = test_img @ test_txt.T
            # Recall@1: 맞는 텍스트가 1위인 비율
            r1_i2t = (scores.argmax(dim=1) == torch.arange(n_pairs - split)).float().mean().item() * 100
            r1_t2i = (scores.T.argmax(dim=1) == torch.arange(n_pairs - split)).float().mean().item() * 100
        print(f"epoch {epoch+1:3d} | loss {total_loss/n_batches:.4f} | R@1 i2t {r1_i2t:.1f}% t2i {r1_t2i:.1f}%")

# --- 검색 데모 ---
print("\n--- cross-modal retrieval ---")
model.eval()
with torch.no_grad():
    all_img_emb = model.image_encoder(image_features)
    all_txt_emb = model.text_encoder(text_features)

    # 텍스트로 이미지 검색
    for query_idx in [split, split+5, split+10]:
        txt_emb = all_txt_emb[query_idx:query_idx+1]
        scores = (txt_emb @ all_img_emb.T).squeeze()
        top3 = scores.topk(3)
        hit = query_idx in top3.indices.tolist()
        print(f"  text[{query_idx}] -> images: {top3.indices.tolist()} {'[HIT]' if hit else '[MISS]'}")
