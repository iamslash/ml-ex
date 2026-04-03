"""
Toxicity Classifier with PyTorch.
텍스트의 유해성을 분류하고 임계값 분석.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter

torch.manual_seed(42)

# --- 합성 데이터 ---
safe_texts = [
    "have a nice day", "great work on this project", "thanks for your help",
    "the weather is beautiful today", "I love this movie", "good morning everyone",
    "this is a wonderful idea", "please pass the salt", "happy birthday to you",
    "the sunset looks amazing", "I appreciate your feedback", "lets work together",
    "congratulations on your achievement", "this book is interesting", "enjoy your meal",
    "the garden looks lovely", "thank you for sharing", "hope you feel better soon",
    "welcome to the team", "have a great weekend",
]

toxic_texts = [
    "you are stupid and worthless", "shut up you fool", "I hate everything about you",
    "this is the worst garbage ever", "you are an absolute moron", "go away nobody likes you",
    "what a terrible disgusting person", "you should be ashamed of yourself",
    "this is completely idiotic", "you are so dumb and pathetic",
    "stop being such a loser", "nobody cares about your opinion",
    "you are the worst person ever", "this is pure trash", "get lost you idiot",
    "what an ugly stupid idea", "you make me sick", "this is horrible and offensive",
    "shut your mouth right now", "you deserve nothing good",
]

texts = safe_texts + toxic_texts
labels = [0] * len(safe_texts) + [1] * len(toxic_texts)

# shuffle
perm = torch.randperm(len(texts)).tolist()
texts = [texts[i] for i in perm]
labels = [labels[i] for i in perm]

split = int(0.8 * len(texts))
train_texts, test_texts = texts[:split], texts[split:]
train_labels, test_labels = labels[:split], labels[split:]

print(f"train: {split}, test: {len(texts)-split}")

# --- 간단한 BoW 인코딩 ---
vocab = {}
for text in texts:
    for word in text.lower().split():
        if word not in vocab:
            vocab[word] = len(vocab)

def encode(text):
    vec = torch.zeros(len(vocab))
    for word in text.lower().split():
        if word in vocab:
            vec[vocab[word]] = 1
    return vec

X_train = torch.stack([encode(t) for t in train_texts])
X_test = torch.stack([encode(t) for t in test_texts])
y_train = torch.tensor(train_labels, dtype=torch.float)
y_test = torch.tensor(test_labels, dtype=torch.float)

print(f"vocab: {len(vocab)}")

# --- 모델 ---
model = nn.Sequential(
    nn.Linear(len(vocab), 32),
    nn.ReLU(),
    nn.Linear(32, 1),
)
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

# --- 학습 ---
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    pred = model(X_train).squeeze()
    loss = criterion(pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        acc = ((pred > 0).float() == y_train).float().mean().item() * 100
        print(f"epoch {epoch+1:3d} | loss {loss.item():.4f} | train acc {acc:.1f}%")

# --- 임계값 분석 ---
print("\n--- threshold analysis ---")
model.eval()
with torch.no_grad():
    test_scores = torch.sigmoid(model(X_test).squeeze())

for threshold in [0.3, 0.5, 0.7, 0.9]:
    preds = (test_scores >= threshold).float()
    tp = ((preds == 1) & (y_test == 1)).sum().item()
    fp = ((preds == 1) & (y_test == 0)).sum().item()
    fn = ((preds == 0) & (y_test == 1)).sum().item()
    tn = ((preds == 0) & (y_test == 0)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    fpr = fp / (fp + tn + 1e-8)

    print(f"  threshold={threshold:.1f} | precision={precision:.2f} recall={recall:.2f} FPR={fpr:.2f} | TP={tp} FP={fp} FN={fn} TN={tn}")

# --- 추론 ---
print("\n--- inference ---")
test_sentences = [
    "you are a wonderful person",
    "shut up you idiot",
    "this project is going well",
    "you are so stupid and ugly",
]

with torch.no_grad():
    for sent in test_sentences:
        score = torch.sigmoid(model(encode(sent).unsqueeze(0)).squeeze()).item()
        label = "TOXIC" if score > 0.5 else "SAFE"
        print(f"  [{label:5s} {score:.3f}] {sent}")
