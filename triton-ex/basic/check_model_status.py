import requests

# Triton 서버 주소
TRITON_URL = "http://localhost:8000/v2/models/resnet18"

def main():
    # 모델 상태 확인
    response = requests.get(TRITON_URL)
    if response.status_code == 200:
        print("Model is registered:", response.json())
    else:
        print("Error:", response.text)

if __name__ == "__main__":
    main()
    