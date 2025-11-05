# app/services/inference.py
from __future__ import annotations
import os
from typing import Optional, Dict, Any, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# .env 에서 바꿀 수 있음
HF_MODEL_ID = os.getenv(
    "HF_MODEL_ID",
    "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B",
).strip()
# (선택) LoRA 어댑터 쓰면 여기 모델 repo/id 넣어두기
USE_LORA_ADAPTER_ID = os.getenv("USE_LORA_ADAPTER_ID", "").strip()

# CPU면 토큰 수를 너무 크게 잡지 말자 (속도 ↑)
GEN_KW = dict(
    max_new_tokens=40,      # 느리면 80~120으로 낮추기
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.2,
    do_sample=False,          
)
STOP_STRINGS = ["<|endofturn|>", "<|stop|>"]   # 모델 템플릿 기준 종료 토큰 후보

class _Holder:
    tok = None
    model = None
    device = (
        "cuda" if torch.cuda.is_available()
        else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
              else "cpu")
    )

    @classmethod
    def load(cls):
     if cls.model is not None:
        return

    # CPU 최적화(스레드 수 조정)
     if cls.device == "cpu":
        try:
            torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        except Exception:
            pass

    # 토크나이저
     cls.tok = AutoTokenizer.from_pretrained(HF_MODEL_ID, use_fast=True)

     if cls.device == "cpu":
        # ✅ CPU에서는 device_map을 'cpu'로 고정하고 dtype은 float32로
        cls.model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_ID,
            device_map={"": "cpu"},
            dtype=torch.float32,

            low_cpu_mem_usage=True,
        )
     else:
        # GPU/MPS일 때만 bfloat16 등 사용
        dtype = torch.bfloat16 if cls.device == "cuda" else None
        cls.model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_ID,
            device_map="auto",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

    if USE_LORA_ADAPTER_ID:
        try:
            from peft import PeftModel
            cls.model = PeftModel.from_pretrained(cls.model, USE_LORA_ADAPTER_ID)
            print(f"[LoRA] Loaded adapter: {USE_LORA_ADAPTER_ID}")
        except Exception as e:
            print(f"[WARN] Failed to load LoRA adapter: {e}")
# app/services/inference.py
def _build_chat(user_text: str) -> list[dict[str, str]]:
    return [
        {"role": "tool_list", "content": ""},
        {"role": "system", "content": (
            "역할: 한국어 공감 코멘트 생성기.\n"
            "규칙:\n"
            "1) 2~3문장으로 공감 요약 후, 1문장으로 작은 제안을 덧붙여라.\n"
            "2) 이모지·해시태그·인용부호·출처 표시 금지.\n"
            "3) 존대하지만 가볍고 따뜻한 톤.\n"
            "4) 사용자가 말하지 않은 사실은 추정하지 말 것.\n"
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
        tok, model = _Holder.tok, _Holder.model

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
        # 보통 템플릿에 assistant 마커가 섞임 → 이후만 취함
        if "<|im_start|>assistant" in decoded:
            decoded = decoded.split("<|im_start|>assistant")[-1]
        return _strip_after_stop(decoded).strip()
