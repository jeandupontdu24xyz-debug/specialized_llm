import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os, json

MODEL = SentenceTransformer("all-MiniLM-L6-v2")
DATA_PATH = "../data/processed/train.jsonl"
INDEX_PATH = "../outputs/vectorstore/faiss.index"

os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

texts = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        texts.append(ex["response"])

embeddings = MODEL.encode(texts, convert_to_numpy=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, INDEX_PATH)

print("Index vectoriel FAISS créé :", INDEX_PATH)
