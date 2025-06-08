import numpy as np
import requests

# 가짜 입력 데이터 생성 (1x3x224x224)
input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)

payload = {
    "inputs": [
        {
            "name": "input__0",
            "shape": list(input_data.shape),
            "datatype": "FP32",
            "data": input_data.flatten().tolist()
        }
    ],
    "outputs": [
        {
            "name": "output__0"
        }
    ]
}

# REST API 요청
url = "http://localhost:8000/v2/models/resnet18/infer"
response = requests.post(url, json=payload)
result = response.json()

print("✅ 추론 응답 결과 shape:", np.array(result["outputs"][0]["data"]).shape)
print("📊 일부 결과:", result["outputs"][0]["data"][:10])
