"""
evaluate_model.py
-----------------
Évaluation automatique d’un modèle LLM fine-tuné ou RLHF.
Calcule les métriques (ROUGE, BLEU, perplexité) et génère un rapport synthétique.
"""

import os
import json
import torch
import evaluate
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# === CONFIGURATION ===
MODEL_PATH = "../models/rlhf"  # ou "../models/fine_tuned"
DATA_PATH = "../data/processed/test.jsonl"
OUTPUT_PATH = "../outputs/metrics/evaluation_report.json"

MAX_SAMPLES = 50  # Pour limiter la charge GPU lors du test
MAX_NEW_TOKENS = 128

# === CHARGEMENT DU MODÈLE ===
print("🔹 Chargement du modèle et du tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

# === CHARGEMENT DU DATASET DE TEST ===
print("🔹 Chargement du dataset de test...")
samples = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= MAX_SAMPLES:
            break
        samples.append(json.loads(line))

print(f"Nombre d’échantillons utilisés : {len(samples)}")

# === ÉVALUATION DES SORTIES ===
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

generated_texts = []
references = []

print("🔹 Génération des réponses...")
for ex in tqdm(samples, desc="Évaluation du modèle"):
    prompt = ex["prompt"]
    reference = ex["response"]

    try:
        output = generator(prompt, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.7)
        text = output[0]["generated_text"].replace(prompt, "").strip()
    except Exception as e:
        text = ""
        print(f"Erreur sur un exemple : {e}")

    generated_texts.append(text)
    references.append(reference)

# === CALCUL DES MÉTRIQUES ===
rouge_result = rouge.compute(predictions=generated_texts, references=references)
bleu_result = bleu.compute(predictions=generated_texts, references=references)

# Optionnel : calcul de la perplexité
print("🔹 Calcul de la perplexité...")
def compute_perplexity(model, tokenizer, texts):
    ppl = []
    for t in texts:
        inputs = tokenizer(t, return_tensors="pt", truncation=True).to(model.device)
        with torch.no_grad():
            loss = model(**inputs, labels=inputs["input_ids"]).loss
        ppl.append(np.exp(loss.item()))
    return float(np.mean(ppl))

perplexity = compute_perplexity(model, tokenizer, references[:20])

# === SYNTHÈSE ===
report = {
    "rouge": rouge_result,
    "bleu": bleu_result,
    "perplexity": perplexity,
    "nb_samples": len(samples)
}

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4, ensure_ascii=False)

print("\n Rapport généré avec succès :", OUTPUT_PATH)
print(json.dumps(report, indent=4, ensure_ascii=False))
