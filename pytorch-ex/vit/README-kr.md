# ViT (Vision Transformer)

> 이미지를 패치로 나누어 Transformer에 입력하는 이미지 분류 모델.

## 실행 방법

```bash
cd pytorch-ex/vit
pip install transformers
python vit.py
```

첫 실행 시 `google/vit-base-patch16-224` 모델을 다운로드한다 (~350MB).

## 핵심 개념

### CNN → ViT

| | CNN | ViT |
|---|---|---|
| 입력 처리 | 합성곱 필터 슬라이딩 | 이미지를 패치로 분할 |
| 관계 학습 | 로컬 (커널 크기) | 글로벌 (어텐션) |
| 사전학습 | ImageNet 분류 | ImageNet 분류 |

### 패치 임베딩

```
224x224 이미지
  → 16x16 패치 196개로 분할
  → 각 패치를 768차원 벡터로 임베딩
  → [CLS] 토큰 추가 → 197개 토큰
  → Transformer 인코더 12개 레이어
  → [CLS] 출력으로 분류
```

### BERT와의 관계

ViT는 사실상 "이미지용 BERT":
- BERT: 단어 → 토큰 임베딩 → Transformer
- ViT: 이미지 패치 → 패치 임베딩 → Transformer
- 둘 다 [CLS] 토큰을 분류에 사용

### Self-Attention과의 관계

scratch-ex/self-attention에서 구현한 어텐션이 ViT 내부에서도 동일하게 동작한다.
차이점은 입력이 단어 임베딩이 아니라 이미지 패치 임베딩이라는 것뿐.
