import requests
import numpy as np
import json

# Triton 서버 주소 및 모델 정보
TRITON_URL = "http://localhost:8000/v2/models/resnet18/versions/1/infer"

def main():
    # (배치, 채널, 높이, 너비) = (1, 3, 224, 224)
    input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)

    # 입력 데이터를 JSON 형식으로 변환
    payload = {
        "inputs": [
            {
                "name": "input__0",
                "shape": input_data.shape,
                "datatype": "FP32",
                "data": input_data.tolist()
            }
        ]
    }

    # 추론 요청
    response = requests.post(TRITON_URL, json=payload)

    # 결과 출력
    if response.status_code == 200:
        result = response.json()
        output = np.array(result["outputs"][0]["data"])
        print("Output shape:", output.shape)
        print("Top 5 indices:", np.argsort(output)[-5:][::-1])
    else:
        print("Error:", response.text)

if __name__ == "__main__":
    main()
