"""
Module pour charger et utiliser les modèles transformers entraînés
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class TransformerFakeNewsDetector:
    def __init__(self, model_path, model_name, language='english'):
        self.language = language
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        
        # Charger les poids entraînés
        if os.path.exists(model_path):
            print(f"Chargement du modèle depuis {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print(f"Fichier de modèle {model_path} non trouvé, utilisation du modèle pré-entraîné")
        
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, text, threshold=0.5):
        # Tokeniser le texte
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=64
        ).to(self.device)
        
        # Faire la prédiction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        # 0 = vrai, 1 = faux (selon notre convention lors de l'entraînement)
        is_fake = prediction == 1
        
        return {
            "is_fake": is_fake,
            "confidence": confidence,
            "probabilities": probabilities[0].cpu().numpy(),
            "language": self.language
        }

# Fonction utilitaire pour choisir le modèle en fonction de la langue
def get_detector_for_language(text, lang=None):
    if lang is None:
        # Détection automatique de la langue (implémentation simplifiée)
        if any(c in text.lower() for c in ['é', 'è', 'ê', 'à', 'ç']):
            lang = 'french'
        else:
            lang = 'english'
    
    if lang == 'french':
        model_path = "models/pytorch/camembert-base_french.pt"
        model_name = "camembert-base"
        return TransformerFakeNewsDetector(model_path, model_name, 'french')
    else:
        model_path = "models/pytorch/distilbert-base-uncased_english.pt"
        model_name = "distilbert-base-uncased"
        return TransformerFakeNewsDetector(model_path, model_name, 'english')

# Exemple d'utilisation
def predict_fake_news(text, lang=None):
    start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    
    if start_time:
        start_time.record()
    
    import time
    time_start = time.time()
    
    detector = get_detector_for_language(text, lang)
    result = detector.predict(text)
    
    if end_time:
        end_time.record()
        torch.cuda.synchronize()
        processing_time = start_time.elapsed_time(end_time) / 1000.0  # Convertir en secondes
    else:
        processing_time = time.time() - time_start
    
    return result["is_fake"], result["probabilities"], processing_time, result["language"]
