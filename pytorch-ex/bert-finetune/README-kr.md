# BERT Fine-tuning

> 사전학습된 BERT 모델을 감성 분류 태스크에 맞게 fine-tuning.

## 실행 방법

```bash
cd pytorch-ex/bert-finetune
pip install transformers
python bert_finetune.py
```

첫 실행 시 `bert-base-uncased` 모델을 다운로드한다 (~440MB).

## 핵심 개념

### BERT란

- **B**idirectional **E**ncoder **R**epresentations from **T**ransformers
- 양방향 Transformer 인코더
- MLM (Masked Language Model)으로 사전학습됨
- [CLS] 토큰의 출력을 분류에 사용

### Fine-tuning 전략

```
[사전학습된 BERT] → [분류 헤드 추가] → [소량 데이터로 학습]
```

이 코드에서는:
- BERT 대부분의 레이어 동결 (freeze)
- 마지막 2개 레이어 + 분류 헤드만 학습
- 전체 파라미터의 ~14%만 학습

### 인코더 vs 디코더

| | BERT (인코더) | GPT (디코더) |
|---|---|---|
| 어텐션 | 양방향 (전체 참조) | 단방향 (이전만 참조) |
| 사전학습 | MLM (빈칸 채우기) | Next Token Prediction |
| 용도 | 분류, NER, QA | 텍스트 생성 |

### scratch Self-Attention과의 관계

scratch-ex/self-attention에서 구현한 것이 BERT 내부의 각 레이어에 들어있다.
- 12개 레이어 x 12개 헤드 = 144개의 어텐션 연산
- Fine-tuning은 이 구조를 그대로 쓰고, 마지막 분류 부분만 조정
