from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

MODEL = "mistralai/Mistral-7B-v0.3"
DATA_PATH = "../data/processed/rlhf_dataset.jsonl"
OUTPUT = "../models/reward_model"

dataset = load_dataset("json", data_files=DATA_PATH, split="train")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token

def preprocess(ex):
    return tokenizer(ex["prompt"] + ex["response"], truncation=True, padding="max_length", max_length=512)
dataset = dataset.map(preprocess)

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1)

args = TrainingArguments(output_dir=OUTPUT, per_device_train_batch_size=2, num_train_epochs=1)
trainer = Trainer(model=model, args=args, train_dataset=dataset)
trainer.train()

model.save_pretrained(OUTPUT)
print("Modèle de récompense entraîné et sauvegardé.")
