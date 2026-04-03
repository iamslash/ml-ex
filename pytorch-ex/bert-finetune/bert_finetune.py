"""
BERT Fine-tuning with PyTorch + Hugging Face.
사전학습된 BERT로 감성 분류 (SST-2 유사 합성 데이터).
"""

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# --- Hugging Face 모델 로드 ---
try:
    from transformers import BertTokenizer, BertModel
except ImportError:
    print("pip install transformers 필요")
    print("  pip install transformers")
    exit(1)

model_name = "bert-base-uncased"
print(f"loading {model_name}...")
tokenizer = BertTokenizer.from_pretrained(model_name)
bert = BertModel.from_pretrained(model_name)
print(f"bert params: {sum(p.numel() for p in bert.parameters()):,}")

# --- 합성 감성 데이터 ---
texts = [
    # 긍정 (1)
    "this movie is great and I love it",
    "wonderful performance and amazing story",
    "brilliant acting and excellent direction",
    "really enjoyed this beautiful film",
    "fantastic movie with perfect cast",
    "absolutely love this masterpiece",
    "outstanding film with great visuals",
    "best movie I have seen this year",
    "incredible story and amazing acting",
    "truly wonderful and inspiring film",
    # 부정 (0)
    "this movie is terrible and boring",
    "awful performance and bad story",
    "worst film I have ever seen",
    "really hated this ugly disaster",
    "horrible movie with terrible cast",
    "absolutely hate this garbage",
    "dreadful film with poor visuals",
    "worst movie I have seen this year",
    "terrible story and awful acting",
    "truly horrible and depressing film",
]
labels = [1]*10 + [0]*10

# 토큰화
encodings = tokenizer(texts, padding=True, truncation=True, max_length=32, return_tensors='pt')
input_ids = encodings['input_ids'].to(device)
attention_mask = encodings['attention_mask'].to(device)
labels_t = torch.tensor(labels).to(device)

print(f"\ndata: {len(texts)} samples")
print(f"max token length: {input_ids.shape[1]}")

# --- 분류 모델 ---
class BertClassifier(nn.Module):
    def __init__(self, bert_model, n_classes=2):
        super().__init__()
        self.bert = bert_model
        # BERT 가중치 동결 (마지막 2개 레이어만 학습)
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, n_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.classifier(cls_output)

model = BertClassifier(bert).to(device)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

# --- 학습 ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(input_ids, attention_mask)
    loss = criterion(output, labels_t)
    loss.backward()
    optimizer.step()

    preds = output.argmax(dim=1)
    acc = (preds == labels_t).float().mean().item() * 100

    if (epoch + 1) % 5 == 0:
        print(f"epoch {epoch+1:2d}/{num_epochs} | loss {loss.item():.4f} | accuracy {acc:.1f}%")

# --- 추론 ---
model.eval()
test_texts = [
    "I really enjoyed this great movie",
    "this was the worst film ever made",
    "amazing and beautiful storytelling",
    "boring and terrible experience",
]

print(f"\n--- inference ---")
with torch.no_grad():
    enc = tokenizer(test_texts, padding=True, truncation=True, max_length=32, return_tensors='pt')
    output = model(enc['input_ids'].to(device), enc['attention_mask'].to(device))
    probs = torch.softmax(output, dim=1)

    for text, prob in zip(test_texts, probs):
        sentiment = "positive" if prob[1] > prob[0] else "negative"
        conf = max(prob[0].item(), prob[1].item())
        print(f"  [{sentiment:8s} {conf:.2f}] {text}")
