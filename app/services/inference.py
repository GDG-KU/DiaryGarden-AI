from __future__ import annotations

from typing import Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class InferenceService:
    """
    파인튜닝된 HyperCLOVAX 모델을 호스팅하는 실제 추론 서비스.
    """

    def __init__(self, model_version: str = "hyperclovax-seed-0.5b-diary-v1") -> None:
        self.model_version = model_version
        
        # TODO: 실제 저장된 모델 경로로 변경
        comment_model_path = "./models/model_comment"
        emotion_model_path = "./models/model_emotion"
        
        # 공통 토크나이저 로드 (베이스 모델)
        base_model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # 1. 코멘트 생성 모델 로드
        self.comment_model = AutoModelForCausalLM.from_pretrained(comment_model_path)
        self.comment_model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.comment_model.eval()

        # 2. 감정 분석 모델 로드
        self.emotion_model = AutoModelForCausalLM.from_pretrained(emotion_model_path)
        self.emotion_model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.emotion_model.eval()

    def generate_response(self, text: str, metadata: Optional[dict[str, Any]] = None) -> str:
        """
        metadata의 'task' 필드를 기반으로 적절한 모델을 호출합니다.
        """
        if metadata is None:
            metadata = {}
            
        # metadata에서 'task'를 가져옵니다. 기본값은 'comment'로 설정.
        task_type = metadata.get("task", "comment")

        try:
            if task_type == "comment":
                # 1. 코멘트 생성 태스크
                prompt = f"instruction: 다음 일기를 읽고 따뜻한 코멘트를 작성해줘.\ninput: {text}\noutput:"
                model_to_use = self.comment_model
            
            elif task_type == "emotion":
                # 2. 감정 분석 태스크
                prompt = f"instruction: 다음 일기에서 느껴지는 주된 감정을 한 단어로 분류해줘.\ninput: {text}\noutput:"
                model_to_use = self.emotion_model
                
            else:
                return f"Error: Unknown task type '{task_type}'"

            # 모델 추론 실행 (토크나이징 및 생성)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model_to_use.device)
            
            # TODO: max_new_tokens, do_sample, temperature 등 생성 옵션 조정 필요
            outputs = model_to_use.generate(**inputs, max_new_tokens=100) 
            
            # 프롬프트를 제외한 실제 생성된 텍스트만 디코딩
            generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return generated_text.strip()

        except Exception as e:
            # 실제 운영 시에는 로깅(logging) 필요
            print(f"Error during inference: {e}")
            raise  # 예외를 다시 발생시켜 API 응답(500)으로 처리되도록 함


inference_service = InferenceService()
