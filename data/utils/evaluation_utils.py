import evaluate
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Charge les métriques Hugging Face
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

# Modèle d’embeddings pour la similarité
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def compute_metrics(references, predictions):
    """
    Calcule plusieurs scores automatiques pour évaluer la qualité du modèle.
    """
    results = {}

    # BLEU
    results["bleu"] = bleu.compute(predictions=predictions, references=[[r] for r in references])["bleu"]

    # ROUGE-L
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    results["rougeL"] = rouge_scores["rougeL"]

    # BERTScore (mesure sémantique)
    bert = bertscore.compute(predictions=predictions, references=references, lang="fr")
    results["bertscore_f1"] = np.mean(bert["f1"])

    # Similarité sémantique par embeddings
    emb_ref = embedder.encode(references, convert_to_tensor=True)
    emb_pred = embedder.encode(predictions, convert_to_tensor=True)
    cosine_scores = cosine_similarity(emb_ref.cpu(), emb_pred.cpu()).diagonal()
    results["semantic_similarity"] = float(np.mean(cosine_scores))

    # Score global (pondéré)
    results["global_score"] = (
        0.2 * results["bleu"] +
        0.3 * results["rougeL"] +
        0.3 * results["bertscore_f1"] +
        0.2 * results["semantic_similarity"]
    )

    return results
