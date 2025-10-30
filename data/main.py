"""
main.py – Orchestrateur complet du pipeline LLM
Description :
    Ce script exécute automatiquement toutes les étapes du pipeline de spécialisation d’un modèle LLM disponible en source ouverte
    (ex. Mistral-7B) : prétraitement, fine-tuning, indexation, RAG, RLHF et évaluation.
"""

import os
import subprocess
import datetime
import logging

# Configuration des chemins principaux
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
LOG_DIR = os.path.join(BASE_DIR, "outputs", "logs")

os.makedirs(LOG_DIR, exist_ok=True)

# Configuration du logging
log_path = os.path.join(LOG_DIR, f"pipeline_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Liste ordonnée des étapes à exécuter
PIPELINE_STEPS = [
    ("Préparation du corpus", "prepare_corpus.py"),
    ("Fine-tuning LoRA", "fine_tune_lora.py"),
    ("Indexation vectorielle RAG", "rag_inference.py"),
    ("Fusion des feedbacks humains", "merge_feedbacks_rlhf.py"),
    ("Entraînement du Reward Model", "train_reward_model.py"),
    ("Évaluation automatique du modèle", "evaluate_model.py"),
]

def run_step(label, script_name):
    """
    Exécute un script Python du dossier /scripts et journalise sa sortie.
    """
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    if not os.path.exists(script_path):
        logging.error(f"Script introuvable : {script_name}")
        return False

    logging.info(f"Démarrage de l’étape : {label}")
    print(f"\n=== 🚀 {label} ===")

    try:
        result = subprocess.run(["python", script_path], capture_output=True, text=True, check=True)
        logging.info(result.stdout)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Erreur lors de l’exécution de {script_name} : {e.stderr}")
        print(f"❌ Erreur : {e.stderr}")
        return False

def main():
    print("\n==============================")
    print("🧠 PIPELINE DE SPÉCIALISATION LLM")
    print("==============================\n")

    for step_label, script_file in PIPELINE_STEPS:
        success = run_step(step_label, script_file)
        if not success:
            print(f"Arrêt du pipeline suite à une erreur dans l’étape : {step_label}")
            break
        else:
            print(f"Étape terminée : {step_label}")

    print("\n Pipeline terminé. Voir le log complet dans :", log_path)

if __name__ == "__main__":
    main()
