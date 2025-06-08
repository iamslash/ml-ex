import triton_python_backend_utils as pb_utils
import torch
import torch.nn as nn
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        # 가중치 초기화: y = 2*x1 + 3*x2 + 1
        self.linear.weight.data = torch.tensor([[2.0, 3.0]])
        self.linear.bias.data = torch.tensor([1.0])

class TritonPythonModel:
    def initialize(self, args):
        """모델 초기화"""
        logger.info("Initializing Linear Regression model version 1 (PyTorch)...")
        self.model = LinearRegressionModel()
        self.model.eval()
        logger.info("Linear Regression model version 1 initialized successfully")

    def execute(self, requests):
        """추론 실행"""
        responses = []
        for request in requests:
            # 입력 데이터 가져오기
            in_0 = pb_utils.get_input_tensor_by_name(request, "input__0")
            input_data = in_0.as_numpy()

            # 입력 데이터를 PyTorch 텐서로 변환
            input_tensor = torch.from_numpy(input_data).float()
            
            # 추론
            with torch.no_grad():
                output = self.model(input_tensor)
            
            # 결과를 numpy 배열로 변환
            output_np = output.numpy()
            
            # 응답 생성
            out_tensor = pb_utils.Tensor("output__0", output_np)
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(inference_response)

        return responses

    def finalize(self):
        """모델 정리"""
        pass 