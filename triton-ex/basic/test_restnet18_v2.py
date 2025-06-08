import numpy as np
import tritonclient.http as httpclient

# Triton 서버 주소 및 모델 정보
TRITON_URL = "localhost:8000"
MODEL_NAME = "resnet18"
MODEL_VERSION = "2"

def main():
    # (배치, 채널, 높이, 너비) = (1, 3, 224, 224)
    input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)

    # Triton 클라이언트 생성
    client = httpclient.InferenceServerClient(url=TRITON_URL)

    # 입력 텐서 생성
    input_tensor = httpclient.InferInput("input__0", input_data.shape, "FP32")
    input_tensor.set_data_from_numpy(input_data)

    # 추론 요청
    response = client.infer(
        MODEL_NAME,
        model_version=MODEL_VERSION,
        inputs=[input_tensor]
    )

    # 결과 출력
    output = response.as_numpy("output__0")
    print("Output shape:", output.shape)
    print("Top 5 indices:", np.argsort(output[0])[-5:][::-1])

if __name__ == "__main__":
    main()
    