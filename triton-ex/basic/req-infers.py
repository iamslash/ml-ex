import numpy as np
import requests
import threading
import time

def infer_request(i):
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

    url = "http://localhost:8000/v2/models/resnet18/infer"
    start = time.time()
    r = requests.post(url, json=payload)
    end = time.time()

    if r.status_code == 200:
        print(f"[Client {i}] ✅ 응답 시간: {end - start:.4f}s, 결과 일부: {r.json()['outputs'][0]['data'][:3]}")
    else:
        print(f"[Client {i}] ❌ 오류: {r.status_code} - {r.text}")

# 10개의 클라이언트 스레드를 동시에 실행
threads = []
for i in range(10):
    t = threading.Thread(target=infer_request, args=(i,))
    t.start()
    threads.append(t)

# 모든 스레드 종료까지 대기
for t in threads:
    t.join()
print("모든 클라이언트 요청 완료.")
