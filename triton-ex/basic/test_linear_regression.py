import requests
import numpy as np
import json

def test_model(version):
    # Triton 서버 주소 및 모델 정보
    TRITON_URL = f"http://localhost:8000/v2/models/linear_regression/versions/{version}/infer"

    # 테스트 데이터 생성: [[1, 2], [3, 4], [5, 6]]
    input_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

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
        print(f"\nVersion {version} Results:")
        print("Input:", input_data)
        print("Output:", output.reshape(-1))
        print("Expected:", 2*input_data[:, 0] + 3*input_data[:, 1] + 1)
    else:
        print(f"Error (Version {version}):", response.text)

def main():
    # 버전 1과 2 모두 테스트
    test_model(1)
    test_model(2)

if __name__ == "__main__":
    main() 