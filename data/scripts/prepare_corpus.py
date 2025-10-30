import os
import json
import random
from tqdm import tqdm

DATA_DIR = "data/raw"
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def chunk_text(text, chunk_size=512):
    """Découpe le texte brut en blocs (chunks)."""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def simulate_prompt_response(chunk):
    """Crée un prompt / réponse simulé à partir d’un chunk."""
    prompt = f"Analyse et résume les informations suivantes : {chunk[:200]}..."
    response = f"Résumé : {chunk[:300]}..."
    return prompt, response

def build_dataset():
    dataset = []
    for fname in tqdm(os.listdir(DATA_DIR)):
        if not fname.endswith((".txt", ".docx")):
            continue
        with open(os.path.join(DATA_DIR, fname), "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        chunks = chunk_text(text)
        for c in chunks:
            prompt, response = simulate_prompt_response(c)
            dataset.append({"prompt": prompt, "response": response})
    return dataset

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    print("Construction du dataset à partir de documents sources...")
    data = build_dataset()
    random.shuffle(data)

    n = len(data)
    train, val, test = data[:int(0.8*n)], data[int(0.8*n):int(0.9*n)], data[int(0.9*n):]

    save_jsonl(train, os.path.join(OUTPUT_DIR, "train.jsonl"))
    save_jsonl(val, os.path.join(OUTPUT_DIR, "val.jsonl"))
    save_jsonl(test, os.path.join(OUTPUT_DIR, "test.jsonl"))

    print(f"Datasets sauvegardés dans {OUTPUT_DIR}/ (train/val/test)")
