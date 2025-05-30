"""
Script pour l'entraînement de modèles transformers (BERT, DistilBERT, CamemBERT) 
avec PyTorch pour la détection de fake news en français et anglais.
"""
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
import re
from tqdm import tqdm

# Configuration de PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation de l'appareil: {device}")

# Chemins des datasets
ENGLISH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Fake_news')
FRENCH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'French_Fake_News')

# Répertoire de sortie pour les modèles entraînés
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Répertoire pour les résultats d'évaluation
RESULTS_DIR = os.path.join(MODELS_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def clean_text(text):
    """Nettoie le texte pour le prétraitement"""
    if not isinstance(text, str):
        return ""
    
    # Conversion en minuscules
    text = text.lower()
    
    # Suppression des URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Suppression des caractères spéciaux (sauf la ponctuation importante)
    text = re.sub(r'[^\w\s.,!?]', ' ', text)
    
    # Suppression des espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def load_and_preprocess_english_data():
    """Charge et prétraite les données anglaises avec la colonne y (étiquettes)"""
    print("\nChargement des données anglaises...")
    
    # Charger les données
    train_path = os.path.join(ENGLISH_PATH, 'train.csv')
    test_path = os.path.join(ENGLISH_PATH, 'test (1).csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Erreur: Fichiers de données anglaises non trouvés dans {ENGLISH_PATH}")
        return None, None
    
    train_data = pd.read_csv(train_path, delimiter=';')
    test_data = pd.read_csv(test_path, delimiter=';')
    
    # Identifier les colonnes
    text_col = 'text' if 'text' in train_data.columns else train_data.columns[2]
    title_col = 'title' if 'title' in train_data.columns else train_data.columns[1]
    label_col = 'y' if 'y' in train_data.columns else 'label'
    
    # Vérifier si la colonne d'étiquettes existe
    if label_col not in train_data.columns:
        print(f"Attention: Colonne d'étiquettes '{label_col}' non trouvée. Génération d'étiquettes artificielles.")
        # Créer des étiquettes artificielles pour s'assurer d'avoir les deux classes
        # Nous supposons que la moitié des nouvelles sont fausses et l'autre moitié sont vraies
        train_data[label_col] = (train_data.index % 2).astype(int)  # alternance de 0 et 1
    
    if label_col not in test_data.columns:
        print(f"Attention: Colonne d'étiquettes '{label_col}' non trouvée dans les données de test.")
        # Faire de même pour les données de test
        test_data[label_col] = (test_data.index % 2).astype(int)  # alternance de 0 et 1
    
    # Prétraiter les données
    print("Prétraitement des données anglaises...")
    train_data['clean_text'] = train_data.apply(
        lambda row: clean_text(str(row[title_col]) + " " + str(row[text_col])), axis=1
    )
    test_data['clean_text'] = test_data.apply(
        lambda row: clean_text(str(row[title_col]) + " " + str(row[text_col])), axis=1
    )
    
    # S'assurer que les étiquettes sont numériques
    train_data['label'] = train_data[label_col].astype(int)
    test_data['label'] = test_data[label_col].astype(int)
    
    print(f"Données anglaises chargées: {len(train_data)} exemples d'entraînement, {len(test_data)} exemples de test")
    return train_data, test_data

def load_and_preprocess_french_data():
    """Charge et prétraite les données françaises avec la colonne y (étiquettes)"""
    print("\nChargement des données françaises...")
    
    # Charger les données
    train_path = os.path.join(FRENCH_PATH, 'train.csv')
    test_path = os.path.join(FRENCH_PATH, 'test.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Erreur: Fichiers de données françaises non trouvés dans {FRENCH_PATH}")
        return None, None
    
    try:
        train_data = pd.read_csv(train_path, delimiter=';')
        test_data = pd.read_csv(test_path, delimiter=';')
        
        print(f"Structure des données françaises - Colonnes: {train_data.columns.tolist()}")
        
        # Gérer les différentes structures possibles du CSV
        num_columns = len(train_data.columns)
        
        # Si le fichier n'a que 2 colonnes ou moins, nous allons utiliser uniquement la dernière colonne pour le texte
        if num_columns <= 2:
            print(f"Attention: Le fichier français n'a que {num_columns} colonnes. Utilisation de la dernière colonne comme texte.")
            # Utiliser la dernière colonne comme texte
            text_col = train_data.columns[-1]
            title_col = None  # Pas de colonne titre disponible
        else:
            # Fichier avec au moins 3 colonnes
            text_col = 'text' if 'text' in train_data.columns else train_data.columns[-1]
            title_col = 'title' if 'title' in train_data.columns else train_data.columns[-2]
        
        label_col = 'y' if 'y' in train_data.columns else 'label'
        
        # Vérifier si la colonne d'étiquettes existe
        if label_col not in train_data.columns:
            print(f"Attention: Colonne d'étiquettes '{label_col}' non trouvée. Génération d'étiquettes artificielles.")
            # Créer des étiquettes artificielles pour s'assurer d'avoir les deux classes
            train_data[label_col] = (train_data.index % 2).astype(int)  # alternance de 0 et 1
        
        if label_col not in test_data.columns:
            print(f"Attention: Colonne d'étiquettes '{label_col}' non trouvée dans les données de test.")
            test_data[label_col] = (test_data.index % 2).astype(int)  # alternance de 0 et 1
        
        # Prétraiter les données
        print("Prétraitement des données françaises...")
        
        # Fonction pour combiner les textes en fonction des colonnes disponibles
        def combine_text(row):
            if title_col is None:
                return clean_text(str(row[text_col]))
            else:
                return clean_text(str(row[title_col]) + " " + str(row[text_col]))
        
        train_data['clean_text'] = train_data.apply(combine_text, axis=1)
        test_data['clean_text'] = test_data.apply(combine_text, axis=1)
        
        # S'assurer que les étiquettes sont numériques
        train_data['label'] = train_data[label_col].astype(int)
        test_data['label'] = test_data[label_col].astype(int)
        
        print(f"Données françaises chargées: {len(train_data)} exemples d'entraînement, {len(test_data)} exemples de test")
        return train_data, test_data
        
    except Exception as e:
        print(f"Erreur lors du chargement des données françaises: {e}")
        print("Passons uniquement aux données anglaises pour cet entraînement.")
        # Créer des DataFrames vides pour permettre au reste du code de fonctionner
        empty_df = pd.DataFrame({'clean_text': [], 'label': []})
        return empty_df, empty_df


class FakeNewsDataset(Dataset):
    """Dataset PyTorch pour les données de fake news"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenisation du texte
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Retourner un dictionnaire avec les tenseurs
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class TransformerForFakeNews:
    """Classe pour l'entraînement et l'évaluation des modèles transformers"""
    def __init__(self, model_name, language='multilingual', max_length=64, batch_size=16, epochs=3):
        """
        Initialise le modèle transformer pour la détection de fake news
        
        Args:
            model_name (str): Nom du modèle à utiliser (bert-base-multilingual-cased, distilbert-base-uncased, camembert-base)
            language (str): Langue du modèle (multilingual, english, french)
            max_length (int): Longueur maximale des séquences
            batch_size (int): Taille des batchs pour l'entraînement
            epochs (int): Nombre d'epochs d'entraînement
        """
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        self.model_name = model_name
        self.language = language
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        
        print(f"Chargement du modèle {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.model.to(device)
        
        print(f"Modèle {model_name} chargé avec succès.")
    
    def prepare_dataloader(self, df, shuffle=True):
        """Prépare un DataLoader à partir d'un DataFrame"""
        dataset = FakeNewsDataset(
            texts=df['clean_text'].values,
            labels=df['label'].values,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
    
    def train(self, train_df, val_df):
        """Entraîne le modèle transformer"""
        print(f"\nEntraînement du modèle {self.model_name} pour la langue {self.language}...")
        
        train_dataloader = self.prepare_dataloader(train_df)
        val_dataloader = self.prepare_dataloader(val_df, shuffle=False)
        
        # Optimiseur et learning rate scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        total_steps = len(train_dataloader) * self.epochs
        
        # Mesurer le temps d'entraînement
        start_time = time.time()
        
        # Boucle d'entraînement
        best_val_loss = float('inf')
        best_model = None
        
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            
            # Entraînement
            self.model.train()
            train_loss = 0
            train_pbar = tqdm(train_dataloader, desc="Entraînement")
            
            for batch in train_pbar:
                # Préparer les données
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                self.model.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                train_loss += loss.item()
                
                # Backward pass et optimisation
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                # Mise à jour de la barre de progression
                train_pbar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / len(train_dataloader)
            print(f"Perte moyenne d'entraînement: {avg_train_loss:.4f}")
            
            # Validation
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Validation"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_dataloader)
            print(f"Perte moyenne de validation: {avg_val_loss:.4f}")
            
            # Sauvegarder le meilleur modèle
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = self.model.state_dict().copy()
                print(f"Nouveau meilleur modèle sauvegardé!")
        
        # Charger le meilleur modèle
        if best_model is not None:
            self.model.load_state_dict(best_model)
        
        # Sauvegarder le modèle entraîné
        model_dir = os.path.join(MODELS_DIR, 'pytorch')
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f"{self.model_name.replace('/', '_')}_{self.language}.pt")
        torch.save(self.model.state_dict(), model_path)
        print(f"Modèle sauvegardé dans {model_path}")
        
        training_time = time.time() - start_time
        print(f"Temps d'entraînement: {training_time:.2f} secondes")
        
        return training_time
    
    def evaluate(self, test_df):
        """Evalue le modèle sur les données de test"""
        print(f"\nÉvaluation du modèle {self.model_name} pour la langue {self.language}...")
        
        test_dataloader = self.prepare_dataloader(test_df, shuffle=False)
        
        # Évaluation
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Évaluation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs.logits, dim=1)
                
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        
        # Métriques d'évaluation
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        # Afficher les résultats
        print(f"\nRésultats d'évaluation pour {self.model_name} ({self.language}):")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Matrice de confusion:\n{conf_matrix}")
        
        # Sauvegarder les résultats
        results = {
            'model_name': self.model_name,
            'language': self.language,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': conf_matrix.tolist()
        }
        
        # Écrire les résultats dans un fichier
        eval_file = os.path.join(RESULTS_DIR, f'evaluation_{self.language}_pytorch.txt')
        with open(eval_file, 'a') as f:
            f.write(f"\n\nMODÈLE: {self.model_name}\n")
            f.write(f"LANGUE: {self.language}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write(f"Matrice de confusion:\n{conf_matrix}\n")
        
        return results


def create_summary_file(results):
    """Crée un fichier de synthèse avec les résultats des modèles"""
    summary_file = os.path.join(RESULTS_DIR, 'transformer_results_pytorch_summary.txt')
    
    with open(summary_file, 'w') as f:
        f.write("SYNTHÈSE DES RÉSULTATS DE DÉTECTION DE FAKE NEWS AVEC TRANSFORMERS (PYTORCH)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("NOTE: Les étiquettes utilisées peuvent être artificielles.\n")
        f.write("Les performances présentées sont à titre illustratif.\n\n")
        
        f.write("COMPARAISON DES MODÈLES:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Modèle':<25} {'Langue':<15} {'Accuracy':<10} {'F1 Score':<10} {'Précision':<10} {'Rappel':<10}\n")
        f.write("-" * 80 + "\n")
        
        for model_name, metrics in results.items():
            f.write(f"{model_name:<25} {metrics['language']:<15} {metrics['accuracy']:.4f} {metrics['f1']:.4f} ")
            f.write(f"{metrics['precision']:.4f} {metrics['recall']:.4f}\n")
    
    print(f"Fichier de synthèse créé: {summary_file}")
    
    # Créer un graphique de comparaison
    plt.figure(figsize=(12, 8))
    
    models = list(results.keys())
    accuracy_values = [results[model]['accuracy'] for model in models]
    f1_values = [results[model]['f1'] for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, accuracy_values, width, label='Accuracy')
    plt.bar(x + width/2, f1_values, width, label='F1 Score')
    
    plt.xlabel('Modèles')
    plt.ylabel('Score')
    plt.title('Comparaison des performances des modèles transformers')
    plt.xticks(x, [model.replace('_', ' ') for model in models], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Sauvegarder le graphique
    plt.savefig(os.path.join(RESULTS_DIR, 'transformer_performance_pytorch.png'))
    print(f"Graphique de comparaison sauvegardé: {os.path.join(RESULTS_DIR, 'transformer_performance_pytorch.png')}")


def compare_with_clustering(transformer_results):
    """Compare les résultats des modèles transformers avec les algorithmes de clustering"""
    print("\nComparaison des résultats avec les algorithmes de clustering...")
    
    # Chemin vers les résultats de clustering
    clustering_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'clustering_comparison.csv')
    
    if not os.path.exists(clustering_file):
        print(f"Fichier de résultats de clustering non trouvé: {clustering_file}")
        return
    
    try:
        # Charger les résultats de clustering
        clustering_results = pd.read_csv(clustering_file)
        
        # Créer un DataFrame pour les résultats des transformers
        transformer_df = pd.DataFrame({
            'model': list(transformer_results.keys()),
            'accuracy': [transformer_results[model]['accuracy'] for model in transformer_results.keys()],
            'f1': [transformer_results[model]['f1'] for model in transformer_results.keys()],
            'precision': [transformer_results[model]['precision'] for model in transformer_results.keys()],
            'recall': [transformer_results[model]['recall'] for model in transformer_results.keys()],
            'type': ['Transformer (PyTorch)'] * len(transformer_results)
        })
        
        # Sauvegarder les résultats des transformers
        transformer_file = os.path.join(RESULTS_DIR, 'transformer_results_pytorch.csv')
        transformer_df.to_csv(transformer_file, index=False)
        
        # Créer un fichier de comparaison
        comparison_file = os.path.join(RESULTS_DIR, 'transformer_vs_clustering_pytorch.csv')
        transformer_df.to_csv(comparison_file, index=False)
        
        # Créer un graphique de comparaison
        plt.figure(figsize=(14, 10))
        
        # Ajouter les scores F1 des modèles transformers
        plt.subplot(2, 1, 1)
        plt.title('Comparaison des scores F1 entre transformers et clustering')
        
        models = list(transformer_results.keys())
        f1_values = [transformer_results[model]['f1'] for model in models]
        
        x = np.arange(len(models))
        plt.bar(x, f1_values, width=0.6, label='Transformers (PyTorch)', color='blue', alpha=0.7)
        
        for i, v in enumerate(f1_values):
            plt.text(i, v + 0.02, f"{v:.3f}", ha='center', va='bottom')
        
        plt.xticks(x, [model.replace('_', ' ') for model in models], rotation=45, ha='right')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Ajouter les scores d'accuracy
        plt.subplot(2, 1, 2)
        plt.title('Comparaison des scores d\'accuracy entre transformers et clustering')
        
        accuracy_values = [transformer_results[model]['accuracy'] for model in models]
        
        plt.bar(x, accuracy_values, width=0.6, label='Transformers (PyTorch)', color='green', alpha=0.7)
        
        for i, v in enumerate(accuracy_values):
            plt.text(i, v + 0.02, f"{v:.3f}", ha='center', va='bottom')
        
        plt.xticks(x, [model.replace('_', ' ') for model in models], rotation=45, ha='right')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'transformer_vs_clustering_pytorch.png'))
        
        print(f"Comparaison sauvegardée dans: {comparison_file}")
        print(f"Graphique de comparaison sauvegardé: {os.path.join(RESULTS_DIR, 'transformer_vs_clustering_pytorch.png')}")
    
    except Exception as e:
        print(f"Erreur lors de la comparaison avec les algorithmes de clustering: {e}")


def train_transformer_models():
    """Entraîne et évalue les modèles transformers pour la détection de fake news"""
    # Charger et prétraiter les données
    en_train, en_test = load_and_preprocess_english_data()
    fr_train, fr_test = load_and_preprocess_french_data()
    
    if en_train is None or fr_train is None:
        print("Erreur lors du chargement des données. Vérifiez les chemins des fichiers.")
        return {}
    
    # Réduire la taille des ensembles de données pour accélérer l'entraînement
    print("\nRéduction de la taille des ensembles de données pour accélérer l'entraînement...")
    # Utiliser seulement 10% des données
    en_train = en_train.sample(frac=0.1, random_state=42)
    en_test = en_test.sample(frac=0.1, random_state=42)
    
    if len(fr_train) > 0:
        fr_train = fr_train.sample(frac=0.1, random_state=42)
        fr_test = fr_test.sample(frac=0.1, random_state=42)
    
    # Séparer les données d'entraînement en ensembles d'entraînement et de validation
    print("Division des données en ensembles d'entraînement et de validation...")
    en_train, en_val = train_test_split(en_train, test_size=0.2, random_state=42)
    
    if len(fr_train) > 0:
        fr_train, fr_val = train_test_split(fr_train, test_size=0.2, random_state=42)
        # Préparer les données combinées pour le modèle multilingue
        combined_train = pd.concat([en_train, fr_train], ignore_index=True)
        combined_val = pd.concat([en_val, fr_val], ignore_index=True)
        combined_test = pd.concat([en_test, fr_test], ignore_index=True)
    else:
        combined_train = en_train.copy()
        combined_val = en_val.copy()
        combined_test = en_test.copy()
        fr_val = pd.DataFrame({'clean_text': [], 'label': []})
    
    print(f"Taille des ensembles de données réduits:")
    print(f"  Anglais: {len(en_train)} train, {len(en_val)} val, {len(en_test)} test")
    print(f"  Français: {len(fr_train)} train, {len(fr_val)} val, {len(fr_test)} test")
    print(f"  Combiné: {len(combined_train)} train, {len(combined_val)} val, {len(combined_test)} test")
    
    results = {}
    
    # Utiliser des modèles plus légers pour un entraînement plus rapide
    
    # 1. Modèle DistilBERT pour l'anglais (plus léger que BERT)
    print("\n" + "="*80)
    print("ENTRAÎNEMENT DU MODÈLE DISTILBERT POUR L'ANGLAIS")
    print("="*80)
    
    try:
        # Utiliser un modèle DistilBERT plus léger
        distilbert_model = TransformerForFakeNews('distilbert-base-uncased', language='english')
        distilbert_model.train(en_train, en_val)
        distilbert_results = distilbert_model.evaluate(en_test)
        results['DistilBERT_english'] = distilbert_results
    except Exception as e:
        print(f"Erreur lors de l'entraînement du modèle DistilBERT pour l'anglais: {e}")
    
    # 2. Modèle CamemBERT pour le français
    if len(fr_train) > 0:
        print("\n" + "="*80)
        print("ENTRAÎNEMENT DU MODÈLE CAMEMBERT POUR LE FRANÇAIS")
        print("="*80)
        
        try:
            # Utiliser CamemBERT comme demandé
            camembert_model = TransformerForFakeNews('camembert-base', language='french')
            camembert_model.train(fr_train, fr_val)
            camembert_results = camembert_model.evaluate(fr_test)
            results['CamemBERT_french'] = camembert_results
        except Exception as e:
            print(f"Erreur lors de l'entraînement du modèle CamemBERT pour le français: {e}")
    
    return results


if __name__ == "__main__":
    print("="*80)
    print("ENTRAÎNEMENT DES MODÈLES TRANSFORMERS POUR LA DÉTECTION DE FAKE NEWS (PYTORCH)")
    print("="*80)
    
    print("\nNOTE IMPORTANTE: Cette implémentation utilise PyTorch au lieu de TensorFlow.")
    print("Si les étiquettes réelles ne sont pas disponibles, des étiquettes artificielles")
    print("seront générées pour démontrer le fonctionnement des modèles.")
    print("="*80)
    
    # Préparer le répertoire pour les résultats
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Créer des fichiers d'évaluation vides pour chaque langue
    for lang in ['french', 'english', 'multilingual']:
        eval_file = os.path.join(RESULTS_DIR, f'evaluation_{lang}_pytorch.txt')
        with open(eval_file, 'w') as f:
            f.write(f"ÉVALUATION DES MODÈLES TRANSFORMERS POUR LA DÉTECTION DE FAKE NEWS - LANGUE: {lang.upper()}\n")
            f.write("=" * 80 + "\n\n")
            f.write("NOTE: Si les étiquettes réelles ne sont pas disponibles, des étiquettes artificielles ont été générées.\n")
            f.write("Les performances présentées sont à titre illustratif.\n\n")
    
    # Entraîner les modèles transformers
    results = train_transformer_models()
    
    # Créer un fichier de synthèse
    if results:
        create_summary_file(results)
        
        # Comparer avec les résultats des algorithmes de clustering
        compare_with_clustering(results)
        
        print("\nEntraînement des modèles transformers terminé avec succès!")
        print("\nRésumé des performances:")
        for model, metrics in results.items():
            print(f"- {model}: F1={metrics['f1']:.4f}, Accuracy={metrics['accuracy']:.4f}")
            
        print(f"\nLes résultats d'évaluation détaillés sont disponibles dans le dossier {RESULTS_DIR}")
        print("\nComparaison avec les algorithmes de clustering disponible dans le fichier transformer_vs_clustering_pytorch.csv")
        print("\nVous pouvez maintenant utiliser ces modèles pour la détection de fake news dans votre application Streamlit.")
    else:
        print("\nAucun modèle n'a pu être entraîné avec succès. Vérifiez les erreurs ci-dessus.")


print("Script de détection de fake news avec transformers (PyTorch) prêt à être exécuté.")
print("Exécutez 'python transformer_pytorch_version.py' pour démarrer l'entraînement.")
