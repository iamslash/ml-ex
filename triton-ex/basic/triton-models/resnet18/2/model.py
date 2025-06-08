import triton_python_backend_utils as pb_utils
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import logging
import sys
import numpy as np

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

class TritonPythonModel:
    def initialize(self, args):
        """모델 초기화"""
        try:
            logger.info("Initializing ResNet18 model version 2...")
            self.model = models.resnet18(pretrained=True)
            self.model.eval()
            
            # 이미지 전처리를 위한 transform 정의
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            logger.info("ResNet18 model version 2 initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise

    def preprocess(self, input_tensor):
        """입력 데이터 전처리"""
        try:
            # 입력 데이터가 [B, C, H, W] 형태인지 확인
            if len(input_tensor.shape) != 4:
                raise ValueError(f"Expected 4D input tensor, got shape {input_tensor.shape}")
            
            # 입력 데이터가 [0, 1] 범위에 있는지 확인하고 정규화
            if input_tensor.max() > 1.0 or input_tensor.min() < 0.0:
                input_tensor = input_tensor / 255.0
            
            # PyTorch 텐서로 변환
            input_tensor = torch.FloatTensor(input_tensor)
            
            # 채널 순서가 [B, C, H, W]인지 확인
            if input_tensor.shape[1] != 3:
                raise ValueError(f"Expected 3 channels, got {input_tensor.shape[1]}")
            
            return input_tensor
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise

    def execute(self, requests):
        """추론 실행"""
        responses = []
        
        for request in requests:
            try:
                # 입력 데이터 가져오기
                in_0 = pb_utils.get_input_tensor_by_name(request, "input__0")
                input_tensor = in_0.as_numpy()
                
                logger.info(f"Processing request with input shape: {input_tensor.shape}")
                
                # 전처리
                input_tensor = self.preprocess(input_tensor)
                
                # 추론 실행
                with torch.no_grad():
                    output = self.model(input_tensor)
                
                logger.info(f"Inference completed. Output shape: {output.shape}")
                
                # 출력 텐서 생성
                out_tensor = pb_utils.Tensor("output__0", output.numpy())
                inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
                responses.append(inference_response)
                
            except Exception as e:
                logger.error(f"Error processing request: {str(e)}")
                error = pb_utils.TritonError(f"Error processing request: {str(e)}")
                responses.append(pb_utils.InferenceResponse(error=error))
            
        return responses

    def finalize(self):
        """모델 정리"""
        logger.info("Finalizing ResNet18 model version 2...")
        self.model = None 