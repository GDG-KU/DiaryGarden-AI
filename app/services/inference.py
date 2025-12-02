# app/services/inference.py
from __future__ import annotations
from pydantic import BaseModel
import os
from typing import Optional, Dict, Any, List
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
import re
from fastapi import APIRouter

# ==========================================
# 1. 데이터 모델 정의
# ==========================================
class EmotionResult(BaseModel):
    emotion: str  # 감정 이름 
    intensity: float # 감정 강도 
    model_version: str = "v1.0.0_placeholder"

class InferenceInput(BaseModel):
    text: str

# ==========================================
# 2. 라우터 및 환경 변수 설정
# ==========================================
router = APIRouter(
    prefix="/inference",  
    tags=["Comment Generation", "Emotion Classification"]
)

# .env 또는 기본값 설정
COMMENT_MODEL_ID = os.getenv("HF_MODEL_ID", "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B").strip()

# 감정 모델 경로 (프로젝트 루트 기준)
EMOTION_MODEL_ID = "train/emotion-model" 

# 생성 옵션
GEN_KW = dict(
    max_new_tokens=70,
    do_sample=True,
    temperature=1.5,
    top_p=0.85,
    repetition_penalty=1.2,
    no_repeat_ngram_size=4,
)

STOP_STRINGS = ["<|endofturn|>", "<|stop|>", "<|im_end|>"]

# ==========================================
# 3. 유틸리티 함수
# ==========================================
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
            "7) 자기 자신이 아닌 일기의 본인인 타인에 대한 코멘트를 말해야 한다.\n"
        )},
        {"role": "user", "content": user_text},
    ]

# ==========================================
# 4. 모델 관리 클래스 (_Holder)
# ==========================================
class _Holder:
    # 모델 변수 초기화
    tok: Optional[AutoTokenizer] = None
    comment_model: Optional[AutoModelForCausalLM] = None 
    emotion_model: Optional[AutoModelForSequenceClassification] = None 
    
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

        # CPU 최적화
        if cls.device == "cpu":
            try:
                torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
            except Exception:
                pass

        # 2. 토크나이저 로드 (KoELECTRA 기준)
        if cls.tok is None:
             cls.tok = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator", use_fast=True)

        # 3. 코멘트 생성 모델 로드
        if cls.comment_model is None:
            print(f"Loading comment model: {COMMENT_MODEL_ID}...")
            cls._load_model_for_device(model_id=COMMENT_MODEL_ID, model_type='comment')

        # 4. 감정 분류 모델 로드
        if cls.emotion_model is None:
            print(f"Loading emotion model: {EMOTION_MODEL_ID}...")
            cls._load_model_for_device(model_id=EMOTION_MODEL_ID, model_type='emotion')

        print("Model loading complete.")
    
    @classmethod
    def _load_model_for_device(cls, model_id: str, model_type: str):
        dtype = None
        if cls.device == "cuda":
            dtype = torch.bfloat16
        
        load_func = AutoModelForCausalLM if model_type == 'comment' else AutoModelForSequenceClassification
        target_model_var = 'comment_model' if model_type == 'comment' else 'emotion_model'
        
        try:
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
        except Exception as e:
            print(f"Error loading {model_type} model from {model_id}: {e}")
            raise e

# ==========================================
# 5. 서비스 클래스 (Generator / Classifier)
# ==========================================
class CommentGenerator:
    @classmethod
    async def generate_comment(cls, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        _Holder.load()
        # [수정됨] cls.tok 대신 _Holder.tok 사용
        tok, model = _Holder.tok, _Holder.comment_model

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
            decoded = _to_one_sentence(decoded)
            return decoded

class EmotionClassifier:
    @classmethod
    async def analyze_emotion(cls, text: str, metadata: Optional[Dict[str, Any]] = None) -> EmotionResult:
        _Holder.load() 
        # [수정됨] cls.tok 대신 _Holder.tok 사용
        tok, model = _Holder.tok, _Holder.emotion_model
    
        inputs = tok(
            text,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1).squeeze().tolist() 
                    
        if isinstance(probabilities, float): 
            probabilities = [probabilities]

        max_prob = max(probabilities)
        predicted_index = probabilities.index(max_prob)
    
        emotion_label = model.config.id2label[predicted_index]
    
        return EmotionResult(
            emotion=emotion_label,
            intensity=round(max_prob, 4),
            model_version=EMOTION_MODEL_ID.split("/")[-1]
        )

# ==========================================
# 6. API 엔드포인트
# ==========================================
@router.post("/emotion", response_model=EmotionResult, tags=["Emotion Classification"])
async def analyze_emotion_api(data: InferenceInput):
    """
    일기 텍스트를 받아 감정 분류 결과를 반환하는 API 엔드포인트입니다.
    """
    result = await EmotionClassifier.analyze_emotion(data.text)
    return result