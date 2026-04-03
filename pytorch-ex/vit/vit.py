"""
ViT (Vision Transformer) with PyTorch.
사전학습된 ViT로 이미지 분류.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# --- Hugging Face ViT 로드 ---
try:
    from transformers import ViTForImageClassification, ViTImageProcessor
except ImportError:
    print("pip install transformers 필요")
    exit(1)

model_name = "google/vit-base-patch16-224"
print(f"loading {model_name}...")
image_processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

total_params = sum(p.numel() for p in model.parameters())
print(f"params: {total_params:,}")

# --- ViT의 구조 이해 ---
print(f"\n--- ViT architecture ---")
print(f"image size: {image_processor.size}")
print(f"patch size: 16x16")
print(f"num patches: {(224//16)**2} = {(224//16)}x{(224//16)}")
print(f"hidden size: {model.config.hidden_size}")
print(f"num layers: {model.config.num_hidden_layers}")
print(f"num heads: {model.config.num_attention_heads}")
print(f"num classes: {model.config.num_labels}")

# --- 합성 이미지로 추론 ---
print(f"\n--- inference with synthetic images ---")

def make_synthetic_image(pattern="horizontal"):
    """224x224 RGB 합성 이미지 생성"""
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    if pattern == "horizontal":
        for i in range(0, 224, 32):
            img[i:i+16, :, :] = 255
    elif pattern == "vertical":
        for j in range(0, 224, 32):
            img[:, j:j+16, :] = 255
    elif pattern == "diagonal":
        for i in range(224):
            j = i
            img[max(0,i-8):min(224,i+8), max(0,j-8):min(224,j+8), :] = 255
    elif pattern == "solid_red":
        img[:, :, 0] = 255
    return img

patterns = ["horizontal", "vertical", "diagonal", "solid_red"]

model.eval()
with torch.no_grad():
    for pattern in patterns:
        img = make_synthetic_image(pattern)
        inputs = image_processor(images=img, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        pred_id = logits.argmax(-1).item()
        pred_label = model.config.id2label[pred_id]
        confidence = torch.softmax(logits, dim=-1)[0, pred_id].item()
        print(f"  {pattern:12s} -> {pred_label} ({confidence:.3f})")

# --- 패치 임베딩 시각화 ---
print(f"\n--- patch embedding analysis ---")
with torch.no_grad():
    img = make_synthetic_image("horizontal")
    inputs = image_processor(images=img, return_tensors="pt")
    outputs = model.vit(**inputs, output_hidden_states=True)

    # [CLS] + 196 패치 토큰
    last_hidden = outputs.last_hidden_state  # (1, 197, 768)
    cls_token = last_hidden[0, 0]     # (768,)
    patch_tokens = last_hidden[0, 1:]  # (196, 768)

    print(f"CLS token shape: {cls_token.shape}")
    print(f"patch tokens shape: {patch_tokens.shape}")
    print(f"CLS token norm: {cls_token.norm():.2f}")
    print(f"patch tokens mean norm: {patch_tokens.norm(dim=1).mean():.2f}")
