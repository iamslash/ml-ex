#!/bin/bash

set -e

MODEL_NAME="resnet18"
MODEL_DIR="triton-models/${MODEL_NAME}"
VERSION_DIR="${MODEL_DIR}/1"

echo "🚀 PyTorch ResNet18 모델을 TorchScript로 저장합니다..."

# 디렉토리 생성
mkdir -p "${VERSION_DIR}"

# Python 코드로 TorchScript 모델 저장
python3 - <<EOF
import torch
import torchvision.models as models

model = models.resnet18(pretrained=True)
model.eval()

example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

traced_model.save("${VERSION_DIR}/model.pt")
print("✅ TorchScript 모델 저장 완료: ${VERSION_DIR}/model.pt")
EOF

# Triton config.pbtxt 생성
cat <<EOC > "${MODEL_DIR}/config.pbtxt"
name: "${MODEL_NAME}"
platform: "pytorch_libtorch"
max_batch_size: 1
input [
  {
    name: "input__0"
    data_type: TYPE_FP32
    dims: [3, 224, 224]
  }
]
output [
  {
    name: "output__0"
    data_type: TYPE_FP32
    dims: [1000]
  }
]
EOC

echo "✅ config.pbtxt 생성 완료: ${MODEL_DIR}/config.pbtxt"
echo "📁 Triton 모델 디렉토리 구조가 준비되었습니다: triton-models/${MODEL_NAME}/"
