# Toxicity Classifier (유해성 분류)

> 텍스트의 유해성을 분류하고 임계값에 따른 정밀도/재현율 트레이드오프를 분석.

## 실행 방법

```bash
cd safety-ex/toxicity-classifier
python toxicity_classifier.py
```

## 핵심 개념

### 임계값 분석

| 임계값 낮음 (0.3) | 임계값 높음 (0.9) |
|---|---|
| Recall 높음 (유해 콘텐츠 잘 잡음) | Precision 높음 (오탐 적음) |
| FPR 높음 (정상도 차단) | FNR 높음 (유해 놓침) |
| 안전 우선 | 자유 우선 |

### 실무에서의 중요성

안전 시스템에서는 precision/recall 트레이드오프가 핵심. 임계값 하나로 사용자 경험이 크게 달라진다.
