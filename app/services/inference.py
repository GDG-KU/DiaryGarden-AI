
from __future__ import annotations
import os
from typing import Optional, Dict, Any, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_MODEL_ID = os.getenv("HF_MODEL_ID", "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B")
USE_LORA_ADAPTER_ID = os.getenv("USE_LORA_ADAPTER_ID", "").strip()

# CPU 기준으로 속도/품질 밸런스
GEN_KW = dict(
    max_new_tokens=160,      # 느리면 120~140으로 낮추기
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.2,  # 모델 카드 권장
    do_sample=True,
)

STOP_STRINGS = ["<|endofturn|>", "<|stop|>"]

class _Holder:
    tok = None
    model = None
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    @classmethod
    def load(cls):
        if cls.model is not None:
            return
        # CPU일 때 스레드 약간 제한해서 안정성↑
        if cls.device == "cpu":
            try:
                torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
            except Exception:
                pass

        cls.tok = AutoTokenizer.from_pretrained(HF_MODEL_ID, use_fast=True)
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

def _build_chat(user_text: str) -> List[Dict[str, str]]:
    # 모델의 chat 템플릿과 호환되는 role 포맷
    return [
        {"role": "tool_list", "content": ""},
        {"role": "system", "content": "너는 사용자 일기를 공감적으로 요약·반영하여 한 단락 한국어 코멘트를 생성한다."},
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

        chat = _build_chat(text)
        inputs = tok.apply_chat_template(
            chat, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(**inputs, **GEN_KW)

        decoded = tok.batch_decode(out, skip_special_tokens=False)[0]
        # assistant 구간만 추출
        if "<|im_start|>assistant" in decoded:
            decoded = decoded.split("<|im_start|>assistant")[-1]
        return _strip_after_stop(decoded).strip()
