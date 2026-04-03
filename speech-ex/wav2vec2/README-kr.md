# wav2vec2 (음성 분류)

> 사전학습된 wav2vec2 모델로 음성 명령어를 분류하는 fine-tuning 예제.

## 주의사항

이 예제는 다음 환경이 필요하다:
- 대용량 데이터 다운로드 (~2GB, Speech Commands dataset)
- `torchaudio`, `transformers` 설치
- GPU 권장 (CPU로는 학습이 매우 느림)

## 알려진 문제

- `data.py`: train subset에 `validation_list.txt`를 사용하고 있어 train/val 의미가 혼동됨
- `infer.py`: `torchaudio.models.Wav2Vec2Model.from_pretrained()`는 현재 torchaudio 버전에서 지원하지 않을 수 있음
- `train.py`: `evaluation_strategy` 파라미터가 최신 transformers에서 `eval_strategy`로 변경됨

이 코드는 참고용이며, 직접 실행하려면 위 문제를 수정해야 한다.

## 핵심 개념

### wav2vec2란

- 음성 파형(waveform)에서 직접 표현(representation)을 학습하는 모델
- 자기 지도 학습(self-supervised)으로 사전학습
- fine-tuning으로 음성 인식, 분류 등에 활용

### BERT와의 관계

| | BERT | wav2vec2 |
|---|---|---|
| 입력 | 텍스트 (토큰) | 음성 (파형) |
| 사전학습 | MLM (마스킹) | 마스킹된 음성 구간 예측 |
| 활용 | 텍스트 분류, NER | 음성 인식, 명령어 분류 |
