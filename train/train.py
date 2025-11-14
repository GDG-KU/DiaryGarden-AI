from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
import torch

# -------------------------------------------------------------
# 1) 기본 설정
# -------------------------------------------------------------
MODEL_NAME = "monologg/koelectra-base-v3-discriminator"

# 4개 감정 라벨
label2id = {
    "joy": 0,
    "sadness": 1,
    "anger": 2,
    "neutral": 3,
}
id2label = {v: k for k, v in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# -------------------------------------------------------------
# 2) 데이터셋 로드
# -------------------------------------------------------------
# emotion_data.json 위치: 프로젝트 루트 기준
dataset = load_dataset("json", data_files="emotion_data.json")

# 자동 train/test split (8:2)
dataset = dataset["train"].train_test_split(test_size=0.2)

# -------------------------------------------------------------
# 3) 전처리 함수
# -------------------------------------------------------------
def preprocess(batch):
    enc = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    enc["labels"] = [label2id[x] for x in batch["label"]]
    return enc

dataset = dataset.map(preprocess, batched=True)

# -------------------------------------------------------------
# 4) 모델 준비
# -------------------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=4,
    label2id=label2id,
    id2label=id2label,
)

# -------------------------------------------------------------
# 5) 학습 설정
# -------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./emotion-model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),   # GPU 있으면 자동 mixed precision
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

# -------------------------------------------------------------
# 6) 학습 시작
# -------------------------------------------------------------
trainer.train()

# -------------------------------------------------------------
# 7) 결과 저장
# -------------------------------------------------------------
model.save_pretrained("./emotion-model")
tokenizer.save_pretrained("./emotion-model")

print("\n🎉 Training Finished! 모델이 './emotion-model'에 저장되었습니다.")
