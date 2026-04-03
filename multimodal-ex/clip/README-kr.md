# CLIP (Contrastive Language-Image Pre-training)

> 이미지와 텍스트를 같은 임베딩 공간에 매핑하여 cross-modal 검색을 가능하게 하는 모델.

## 실행 방법

```bash
cd multimodal-ex/clip
python clip_toy.py
```

## 핵심 개념

### 구조

```
이미지 → Image Encoder → 이미지 임베딩 ─┐
                                          ├→ cosine similarity matrix
텍스트 → Text Encoder → 텍스트 임베딩 ──┘
```

### InfoNCE Loss

```
L = (L_i2t + L_t2i) / 2
```

- L_i2t: 이미지로 맞는 텍스트 찾기
- L_t2i: 텍스트로 맞는 이미지 찾기
- In-batch negative로 효율적 학습

### 활용

- 이미지 검색 (텍스트 → 이미지)
- 제로샷 분류 (라벨 텍스트와 이미지 매칭)
- 멀티모달 임베딩
