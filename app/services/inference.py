# app/services/inference.py
from __future__ import annotations
from pydantic import BaseModel

# 감정 분류 결과를 위한 출력 데이터 모델
class EmotionResult(BaseModel):
    emotion: str  # 감정 이름 
    intensity: float # 감정 강도 
    model_version: str = "v1.0.0_placeholder"
class InferenceInput(BaseModel):
    text: str
import os
from typing import Optional, Dict, Any, List

import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
import re
from fastapi import APIRouter
router = APIRouter(
    prefix="/inference",  
    tags=["Comment Generation", "Emotion Classification"]
)

STOP_STRINGS = ["<|endofturn|>", "<|stop|>", "<|im_end|>"]

def _strip_after_stop(text: str) -> str:
    for s in STOP_STRINGS:
        if s and s in text:
            text = text.split(s)[0]
    return text

def _clean_text(t: str) -> str:
    t = t.replace("\u200b", "")
    t = re.sub(r"저는\s*AI[^.?!\n]*[.?!]?", "", t, flags=re.IGNORECASE)
    t = re.sub(r"I am an AI[^.?!\n]*[.?!]?", "", t, flags=re.IGNORECASE)
    t = t.replace("*", "")
    t = re.sub(r"[\"“”‘’]+", "", t)
    # 공백 정리
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _to_one_sentence(t: str) -> str:
    t = t.replace("\r", " ").replace("\n", " ")
    sents = re.split(r"(?:(?<=[\.!?])\s+|(?<=다\.)\s+|(?<=요\.)\s+)", t)
    for s in sents:
        s = s.strip(" '\"`")
        if s:
            if not re.search(r"[\.!?]$", s):
                s += "."
            return s
    return t if t.endswith(".") else (t + ".")

# .env 에서 바꿀 수 있음
HF_MODEL_ID = os.getenv(
    "HF_MODEL_ID",
    "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B",
).strip()
EMOTION_MODEL_ID = "../train/emotion-model"
# (선택) LoRA 어댑터 쓰면 여기 모델 repo/id 넣어두기
USE_LORA_ADAPTER_ID = os.getenv("USE_LORA_ADAPTER_ID", "").strip()

# CPU면 토큰 수를 너무 크게 잡지 말자 (속도 ↑)


GEN_KW = dict(
    max_new_tokens=70,
    do_sample=True,
    temperature=1.5,
    top_p=0.85,
    repetition_penalty=1.2,
    no_repeat_ngram_size=4,
)


STOP_STRINGS = ["<|endofturn|>", "<|stop|>"]   # 모델 템플릿 기준 종료 토큰 후보

class _Holder:
    # 모델 변수를 두 개로 분리
    tok: Optional[AutoTokenizer] = None
    comment_model: Optional[AutoModelForCausalLM] = None # 코멘트 생성 모델 (HyperCLOVA)
    emotion_model: Optional[AutoModelForSequenceClassification] = None # 감정 분류 모델 (KoELECTRA)
    
    device = (
        "cuda" if torch.cuda.is_available()
        else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
              else "cpu")
    )

    @classmethod
    def load(cls):
        # 1. 이미 두 모델이 모두 로드되었는지 확인
        if cls.tok is not None and cls.comment_model is not None and cls.emotion_model is not None:
            return

        # CPU 최적화(스레드 수 조정)
        if cls.device == "cpu":
            try:
                torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
            except Exception:
                pass

        # 2. 토크나이저 로드 (두 모델이 공유)
        if cls.tok is None:
             # KoELECTRA의 토크나이저를 로드합니다 (데이터 처리 기준)
             cls.tok = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator", use_fast=True)


        # 3. 코멘트 생성 모델 로드 (CausalLM)
        if cls.comment_model is None:
            print(f"Loading comment model: {COMMENT_MODEL_ID}...")
            cls._load_model_for_device(model_id=COMMENT_MODEL_ID, model_type='comment')

        # 4. 감정 분류 모델 로드 (SequenceClassification)
        if cls.emotion_model is None:
            print(f"Loading emotion model: {EMOTION_MODEL_ID}...")
            cls._load_model_for_device(model_id=EMOTION_MODEL_ID, model_type='emotion')

        print("Model loading complete.")
    
    @staticmethod
    def _load_model_for_device(cls, model_id: str, model_type: str):
        # GPU/MPS 설정
        dtype = None
        if cls.device == "cuda":
            dtype = torch.bfloat16
        
        # 로드할 함수 선택: 코멘트면 CausalLM, 감정이면 SequenceClassification
        load_func = AutoModelForCausalLM if model_type == 'comment' else AutoModelForSequenceClassification
        target_model_var = 'comment_model' if model_type == 'comment' else 'emotion_model'
        
        if cls.device == "cpu":
             model = load_func.from_pretrained(
                model_id,
                device_map={"": "cpu"},
                dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
        else:
            model = load_func.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
        
        setattr(cls, target_model_var, model)
# app/services/inference.py
def _build_chat(user_text: str) -> list[dict[str, str]]:
    return [
        {"role": "tool_list", "content": ""},
        {"role": "system", "content": (
            "역할: 따뜻하게 위로하는 친구처럼 한국어 코멘트를 쓰는 작가.\n"
            "규칙:\n"
            "1) 반드시 개행 없이 한 문장으로만 작성한다.\n"
            "2) 한글 기준 45~60자로 자연스럽게 맞춘다.\n"
            "3) 문장 끝은 마침표 하나로 끝내며, ?, !, 따옴표, 이모지, 해시태그 금지.\n"
            "4) AI, 언어모델, 시스템 등 자기 언급 금지.\n"
            "5) 입력 내용의 핵심 감정을 반드시 반영해 공감한다.\n"
            "6) 조언은 강요가 아닌 부드러운 제안 형태로 표현한다.\n"
            "6) 자기 자신이 아닌 일기의 본인인 타인에 대한 코멘트를 말해야 한다.\n"

            "출력 예시:\n"
            "- 너 즐거운 하루였구나, 그 기분 오래 유지되도록 스스로를 칭찬해줘.\n"
            "- 너 오늘 많이 힘들었겠다, 잠시 쉬며 마음을 가볍게 만들어보자.\n"
            "- 너 정말 억울했을 것 같아, 네 감정을 인정하고 천천히 정리해보자.\n"
        )},
        {"role": "user", "content": user_text},
    ]


def _strip_after_stop(text: str) -> str:
    for s in STOP_STRINGS:
        if s and s in text:
            text = text.split(s)[0]
    return text.strip()

class CommentGenerator:
    @classmethod
    async def generate_comment(cls, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        _Holder.load()
        tok, model = cls.tok, cls.comment_model

        # 템플릿 적용 (HyperCLOVAX 계열은 apply_chat_template 권장)
        chat = _build_chat(text)
        inputs = tok.apply_chat_template(
            chat,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(**inputs, **GEN_KW)
            decoded = tok.batch_decode(out, skip_special_tokens=False)[0]
            if "<|im_start|>assistant" in decoded:
                decoded = decoded.split("<|im_start|>assistant")[-1]
            decoded = _clean_text(_strip_after_stop(decoded))
            decoded = _to_one_sentence(decoded)   # ← 최종 한 문장 강제
            return decoded
class EmotionClassifier:
    @classmethod
    async def analyze_emotion(cls, text: str, metadata: Optional[Dict[str, Any]] = None) -> EmotionResult:
        # 1. 모델 로드 (KoELECTRA Sequence Classification을 로드합니다.)
        _Holder.load() 
        tok, model = cls.tok, cls.emotion_model # 감정 모델(emotion_model) 사용
    
        # 2. 전처리 (분류 모델에 적합한 단순 인코딩)
        inputs = tok(
            text,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        # 3. 입력 데이터를 모델 장치(CPU/GPU)로 이동
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # 4. 모델 추론 및 감정 강도 계산
        with torch.no_grad():
            outputs = model(**inputs)
            # Logits (모델이 예측한 원시 점수)
            logits = outputs.logits
            # Softmax를 사용하여 확률로 변환 (0.0 ~ 1.0)
            probabilities = torch.softmax(logits, dim=1).squeeze().tolist() 
                    
        
   

        # 5. 계산 로직 추가 (NameError 해결) 
        max_prob = max(probabilities)
        predicted_index = probabilities.index(max_prob)
    
        # 모델의 config에 저장된 ID to Label 딕셔너리를 사용
        emotion_label = model.config.id2label[predicted_index]
    
        # 6. 최종 결과 반환 (EmotionResult 객체 반환)
   
        return EmotionResult(
            emotion=emotion_label,
            intensity=round(max_prob, 4), # 강도는 최대 확률값 (0.0 ~ 1.0)
            model_version=EMOTION_MODEL_ID.split("/")[-1]
        )

# HyperCLOVA X 모델에 감정 분류를 요청하는 프롬프트
def _build_emotion_prompt(user_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": (
            "역할: 일기 텍스트의 핵심 감정을 4가지(슬픔, 기쁨, 화남, 보통) 중 하나로 분류하는 감정 분석기.\n"
            "규칙:\n"
            "1) 오직 하나의 감정 단어만 출력한다.\n"
            "2) 감정 외의 다른 설명, 문장, 이모지, 마침표 등은 절대 금지한다.\n"
            "출력 예시: 기쁨\n"
            "출력 예시: 슬픔\n"
        )},
        {"role": "user", "content": user_text},
    ]
 



from fastapi import APIRouter, HTTPException 
#from app.api.models import InferenceInput # 입력 모델의 경로가 맞는지 확인합니다.
# (router 객체가 router라고 가정합니다)
@router.post("/emotion", response_model=EmotionResult, tags=["Emotion Classification"])
async def analyze_emotion_api(data: InferenceInput):
    """
    일기 텍스트를 받아 감정 분류 결과를 반환하는 API 엔드포인트입니다.
    """
    # 1. 감정 분석 실행 (EmotionClassifier 사용)
    emotion_label = await EmotionClassifier.analyze_emotion(data.text)
    
    # 2. 강도 추정 (현재 모델은 강도를 직접 주지 않으므로, 1.0으로 가정)
    
    # 3. 데이터 반환
    return EmotionResult(
        emotion=emotion_label,  # 모델이 반환한 감정 단어
        intensity=1.0,          # 임시로 강도는 1.0으로 설정
        # HF_MODEL_ID는 파일 상단에 정의되어 있으므로 사용 가능
        model_version=HF_MODEL_ID.split("/")[-1] 
    )
