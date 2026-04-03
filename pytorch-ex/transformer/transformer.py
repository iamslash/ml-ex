"""
Minimal Encoder-Decoder Transformer with PyTorch.
간단한 번역 과제 (숫자 시퀀스 반전)로 Transformer 구조를 학습.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# --- Multi-Head Attention ---
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        B = query.size(0)

        Q = self.W_q(query).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(key).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(value).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(out)

# --- Feed Forward ---
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)

# --- Encoder Layer ---
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        attn_out = self.attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

# --- Decoder Layer ---
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))
        cross_attn_out = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# --- Transformer ---
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, dropout, pad_idx):
        super().__init__()
        self.pad_idx = pad_idx
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def make_src_mask(self, src):
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        B, tgt_len = tgt.shape
        pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool()
        return pad_mask & causal_mask.unsqueeze(0).unsqueeze(0)

    def encode(self, src, src_mask):
        x = self.dropout(self.pos_enc(self.tok_emb(src)))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_out, src_mask, tgt_mask):
        x = self.dropout(self.pos_enc(self.tok_emb(tgt)))
        for layer in self.decoder_layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.fc_out(x)

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_out = self.encode(src, src_mask)
        return self.decode(tgt, enc_out, src_mask, tgt_mask)

# --- 데이터: 시퀀스 반전 과제 ---
PAD, BOS, EOS = 0, 1, 2
vocab_size = 13  # 0=PAD, 1=BOS, 2=EOS, 3-12=숫자

def make_batch(batch_size, seq_len=4):
    nums = torch.randint(3, vocab_size, (batch_size, seq_len))
    src = torch.cat([torch.full((batch_size, 1), BOS), nums, torch.full((batch_size, 1), EOS)], dim=1)
    reversed_nums = nums.flip(1)
    tgt = torch.cat([torch.full((batch_size, 1), BOS), reversed_nums, torch.full((batch_size, 1), EOS)], dim=1)
    return src.to(device), tgt.to(device)

# --- 모델 생성 ---
model = Transformer(
    vocab_size=vocab_size, d_model=64, n_heads=4, n_layers=2,
    d_ff=128, dropout=0.1, pad_idx=PAD,
).to(device)

params = sum(p.numel() for p in model.parameters())
print(f"params: {params:,}")

# --- 고정 데이터셋 ---
train_src, train_tgt = make_batch(1000, seq_len=4)

# --- 학습 ---
criterion = nn.CrossEntropyLoss(ignore_index=PAD)
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 300

for epoch in range(num_epochs):
    model.train()
    idx = torch.randperm(1000)[:256]
    src, tgt = train_src[idx], train_tgt[idx]

    output = model(src, tgt[:, :-1])
    output = output.reshape(-1, vocab_size)
    target = tgt[:, 1:].reshape(-1)

    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        preds = output.argmax(dim=1)
        acc = (preds[target != PAD] == target[target != PAD]).float().mean().item() * 100
        print(f"epoch {epoch+1:3d}/{num_epochs} | loss {loss.item():.4f} | token acc {acc:.1f}%")

# --- 추론 (greedy decoding) ---
model.eval()
print("\n--- inference ---")

# 테스트는 학습에 없는 새 데이터
test_src, test_tgt = make_batch(5, seq_len=4)

with torch.no_grad():
    src_mask = model.make_src_mask(test_src)
    enc_out = model.encode(test_src, src_mask)

    tgt_input = torch.full((5, 1), BOS, device=device)
    for _ in range(6):
        tgt_mask = model.make_tgt_mask(tgt_input)
        output = model.decode(tgt_input, enc_out, src_mask, tgt_mask)
        next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
        tgt_input = torch.cat([tgt_input, next_token], dim=1)

for i in range(5):
    src_tokens = test_src[i].tolist()
    expected = test_tgt[i].tolist()
    predicted = tgt_input[i].tolist()
    # EOS 이후 자르기
    if EOS in predicted[1:]:
        predicted = predicted[:predicted.index(EOS, 1) + 1]
    src_nums = [t for t in src_tokens if t >= 3]
    exp_nums = [t for t in expected if t >= 3]
    pred_nums = [t for t in predicted if t >= 3]
    match = "OK" if pred_nums == exp_nums else "FAIL"
    print(f"  input: {src_nums} -> expected: {exp_nums} | predicted: {pred_nums} [{match}]")
