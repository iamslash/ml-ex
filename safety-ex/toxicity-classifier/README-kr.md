# Toxicity Classifier (유해성 분류기)

> Bag-of-Words 특성과 이진 분류 신경망으로 텍스트 유해성을 탐지하고, 임계값 분석으로 FPR/FNR 트레이드오프를 시각화.

## 실행 방법

```bash
cd safety-ex/toxicity-classifier
pip install torch
python toxicity_classifier.py
```

**출력 예시:**
```
train: 32, test: 8
vocab: 87
epoch  20 | loss 0.4123 | train acc 87.5%
...
--- threshold analysis ---
  threshold=0.3 | precision=0.83 recall=1.00 FPR=0.20 | TP=5 FP=1 FN=0 TN=4
  threshold=0.5 | precision=0.91 recall=0.91 FPR=0.09 | TP=5 FP=0 FN=0 TN=5
  threshold=0.7 | precision=1.00 recall=0.80 FPR=0.00 | TP=4 FP=0 FN=1 TN=5
  threshold=0.9 | precision=1.00 recall=0.60 FPR=0.00 | TP=3 FP=0 FN=2 TN=5

--- inference ---
  [SAFE  0.023] you are a wonderful person
  [TOXIC 0.987] shut up you idiot
```

## 핵심 개념

### Bag-of-Words (BoW) 인코딩

단어의 순서를 무시하고 등장 여부만 이진 벡터로 표현:

```python
vec = torch.zeros(vocab_size)
for word in text.lower().split():
    if word in vocab:
        vec[vocab[word]] = 1  # 등장 여부만 표시 (빈도 아님)
```

- 어휘 크기만큼의 희소 벡터
- 단순하지만 짧은 유해 문장에 효과적

### 이진 분류 모델

```
입력: BoW 벡터 (vocab_size차원)
   -> Linear(vocab_size, 32) -> ReLU
   -> Linear(32, 1)
   -> BCEWithLogitsLoss (학습 시)
   -> Sigmoid (추론 시, 0~1 확률)
```

`BCEWithLogitsLoss`는 수치 안정성을 위해 Sigmoid를 내부에서 결합:
```
loss = -[y * log(σ(x)) + (1-y) * log(1 - σ(x))]
```

### 임계값 분석 (Threshold Analysis)

분류 임계값에 따른 오류 트레이드오프:

```
임계값 낮음 -> 더 많은 텍스트를 유해로 분류
  - Recall(재현율) 상승: 실제 유해 텍스트 더 많이 탐지
  - FPR 상승: 무해 텍스트를 유해로 잘못 분류 증가

임계값 높음 -> 더 적은 텍스트를 유해로 분류
  - Precision(정밀도) 상승: 유해로 분류된 것의 정확도 증가
  - FNR 상승: 실제 유해 텍스트를 놓치는 비율 증가
```

| 지표 | 공식 | 의미 |
|------|------|------|
| Precision | TP / (TP + FP) | 유해 판정 중 실제 유해 비율 |
| Recall | TP / (TP + FN) | 실제 유해 중 탐지된 비율 |
| FPR | FP / (FP + TN) | 무해 텍스트 중 오탐 비율 |
| FNR | FN / (TP + FN) | 유해 텍스트 중 미탐 비율 |

### 임계값 선택 기준

- **안전 최우선(낮은 임계값, 예: 0.3)**: 유해 텍스트 누락 비용이 높을 때. 오탐 감수.
- **정밀도 최우선(높은 임계값, 예: 0.7~0.9)**: 사용자 경험 방해가 심각할 때. 일부 유해 허용.
- **균형(0.5)**: 일반적 기본값.

## 관련 모델과의 비교

| 모델 | 특성 | 장점 | 단점 |
|------|------|------|------|
| **BoW + MLP (이 예제)** | 단어 등장 여부 | 빠름, 해석 가능 | 문맥/순서 무시 |
| **TF-IDF + Logistic** | 단어 빈도 가중치 | 단순, 빠름 | 문맥 무시 |
| **LSTM/GRU** | 시퀀스 모델 | 순서 파악 | 학습 느림, 긴 문맥 약함 |
| **BERT fine-tuning** | 사전학습 트랜스포머 | 문맥 이해 우수 | 무거움, 비쌈 |
| **Perspective API** | 대규모 사전학습 | 다국어, 강건 | 블랙박스, 비용 발생 |

**실무 고려사항:**
- 레이블 불균형: 실제 환경에서 유해 텍스트는 소수. 클래스 가중치나 오버샘플링 필요.
- 맥락 의존성: "shut up"은 맥락에 따라 무해할 수 있음. BoW는 이를 구분 불가.
- 우회 공격(Adversarial): 철자 변형, 특수문자 삽입 등으로 BoW 우회 가능. 강건한 전처리 필요.
- ROC 곡선과 AUC: 임계값에 독립적인 모델 성능 평가. 여러 임계값에서 TPR/FPR을 종합.
