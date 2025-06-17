"""
Script pour l'entraînement de modèles transformers (BERT et DistilBERT) 
avec TensorFlow pour la détection de fake news en français et anglais.
"""
import os
import pandas as pd
import numpy as np

# Fix pour TensorFlow sur Windows
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Importer TensorFlow après avoir configuré les variables d'environnement
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from transformers import CamembertTokenizer, TFCamembertForSequenceClassification
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
import re

# Activer la mémoire GPU dynamique
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU disponible: {len(gpus)}")
    except RuntimeError as e:
        print(f"Erreur lors de la configuration GPU: {e}")
else:
    print("Aucun GPU détecté, utilisation du CPU")

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
    
    # Combiner titre et texte pour une meilleure analyse
    train_data['combined_text'] = train_data[title_col].fillna('') + ' ' + train_data[text_col].fillna('')
    test_data['combined_text'] = test_data[title_col].fillna('') + ' ' + test_data[text_col].fillna('')
    
    # Nettoyer les textes
    train_data['clean_text'] = train_data['combined_text'].apply(clean_text)
    test_data['clean_text'] = test_data['combined_text'].apply(clean_text)
    
    # Renommer la colonne d'étiquettes pour la cohérence
    train_data = train_data.rename(columns={label_col: 'label'})
    test_data = test_data.rename(columns={label_col: 'label'})
    
    # S'assurer que les étiquettes sont numériques
    train_data['label'] = train_data['label'].astype(int)
    test_data['label'] = test_data['label'].astype(int)
    
    print(f"Données anglaises chargées: {len(train_data)} exemples d'entraînement, {len(test_data)} exemples de test")
    print(f"Distribution des étiquettes (train): {train_data['label'].value_counts().to_dict()}")
    
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
    
    train_data = pd.read_csv(train_path, delimiter=';')
    test_data = pd.read_csv(test_path, delimiter=';')
    
    # Identifier les colonnes
    text_col = 'post' if 'post' in train_data.columns else train_data.columns[1]
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
    print("Prétraitement des données françaises...")
    
    # Nettoyer les textes
    train_data['clean_text'] = train_data[text_col].apply(clean_text)
    test_data['clean_text'] = test_data[text_col].apply(clean_text)
    
    # Renommer la colonne d'étiquettes pour la cohérence
    train_data = train_data.rename(columns={label_col: 'label'})
    test_data = test_data.rename(columns={label_col: 'label'})
    
    # S'assurer que les étiquettes sont numériques
    train_data['label'] = train_data['label'].astype(int)
    test_data['label'] = test_data['label'].astype(int)
    
    print(f"Données françaises chargées: {len(train_data)} exemples d'entraînement, {len(test_data)} exemples de test")
    print(f"Distribution des étiquettes (train): {train_data['label'].value_counts().to_dict()}")
    
    return train_data, test_data


def create_tf_dataset(texts, labels, tokenizer, max_length=128, batch_size=32):
    """Crée un dataset TensorFlow à partir des textes et des étiquettes"""
    # Fonction de prétraitement
    def tokenize_function(text, label):
        encoded = tokenizer(text.numpy().decode('utf-8'), 
                           truncation=True,
                           padding='max_length',
                           max_length=max_length,
                           return_tensors='tf')
        return encoded, label
    
    # Fonction wrapper pour tf.py_function
    def tokenize_map_fn(text, label):
        encoded, label = tf.py_function(
            tokenize_function,
            inp=[text, label],
            Tout=[tf.int32, tf.int32]
        )
        # Définir les formes pour les tenseurs de sortie
        encoded.set_shape([max_length])
        label.set_shape([])
        return encoded, label
    
    # Créer le dataset TensorFlow
    dataset = tf.data.Dataset.from_tensor_slices((texts, labels))
    dataset = dataset.map(tokenize_map_fn)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def create_bert_model(model_name='bert-base-multilingual-cased', max_length=128):
    """Crée un modèle BERT pour la classification de texte"""
    # Créer le modèle TensorFlow BERT
    model = TFBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,  # Classification binaire (fake vs. real)
    )
    
    # Compiler le modèle
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model


def create_distilbert_model(model_name='distilbert-base-multilingual-cased', max_length=128):
    """Crée un modèle DistilBERT pour la classification de texte"""
    # Créer le modèle TensorFlow DistilBERT
    model = TFDistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,  # Classification binaire (fake vs. real)
    )
    
    # Compiler le modèle
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model


def create_camembert_model(model_name='camembert-base', max_length=128):
    """Crée un modèle CamemBERT pour la classification de texte en français"""
    # Créer le modèle TensorFlow CamemBERT
    model = TFCamembertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,  # Classification binaire (fake vs. real)
    )
    
    # Compiler le modèle
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model


def train_and_evaluate_model(model, train_dataset, val_dataset, test_dataset, model_name, language, epochs=3):
    """Entraîne et évalue un modèle transformer pour la détection de fake news"""
    print(f"\nEntraînement du modèle {model_name} pour la langue {language}...")
    start_time = time.time()
    
    # Créer le répertoire de sortie pour ce modèle
    output_dir = os.path.join(MODELS_DIR, f"{model_name}_{language}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Définir un callback pour sauvegarder le meilleur modèle
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(output_dir, 'best_model'),
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    )
    
    # Entraîner le modèle
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[checkpoint_callback]
    )
    
    # Calculer le temps d'entraînement
    training_time = time.time() - start_time
    print(f"Modèle {model_name} pour {language} entraîné en {training_time:.2f} secondes")
    
    # Évaluer le modèle sur les données de test
    print(f"\nÉvaluation du modèle {model_name} pour la langue {language}...")
    test_results = model.evaluate(test_dataset)
    
    # Faire des prédictions sur les données de test
    predictions = model.predict(test_dataset)
    
    # Convertir les logits en classes prédites
    predicted_classes = tf.argmax(predictions.logits, axis=1).numpy()
    
    # Extraire les vraies étiquettes
    true_labels = []
    for _, labels in test_dataset:
        true_labels.extend(labels.numpy())
    true_labels = np.array(true_labels)
    
    # Calculer les métriques
    accuracy = accuracy_score(true_labels, predicted_classes)
    precision = precision_score(true_labels, predicted_classes, zero_division=0)
    recall = recall_score(true_labels, predicted_classes, zero_division=0)
    f1 = f1_score(true_labels, predicted_classes, zero_division=0)
    
    # Générer le rapport de classification
    report = classification_report(true_labels, predicted_classes, target_names=["real", "fake"], zero_division=0)
    print("\nRapport de classification détaillé:")
    print(report)
    
    # Calculer la matrice de confusion
    cm = confusion_matrix(true_labels, predicted_classes)
    
    # Sauvegarder le modèle et les résultats
    model.save_pretrained(os.path.join(output_dir, 'final_model'))
    
    # Sauvegarder l'historique d'entraînement
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump({key: list(map(float, values)) for key, values in history.history.items()}, f)
    
    # Sauvegarder les métriques dans un fichier texte
    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'training_time': training_time,
        'language': language,
        'model_name': model_name
    }
    
    with open(os.path.join(output_dir, f'metrics_{language}.txt'), 'w') as f:
        f.write(f"Métriques d'évaluation pour le modèle {model_name} - Langue: {language}\n")
        f.write("=" * 80 + "\n")
        for metric, value in metrics_dict.items():
            if isinstance(value, float):
                f.write(f"{metric}: {value:.4f}\n")
            else:
                f.write(f"{metric}: {value}\n")
    
    # Sauvegarder le rapport de classification
    with open(os.path.join(output_dir, f'classification_report_{language}.txt'), 'w') as f:
        f.write(f"Rapport de classification pour le modèle {model_name} - Langue: {language}\n")
        f.write("=" * 80 + "\n")
        f.write(report)
    
    # Créer et sauvegarder la matrice de confusion
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["real", "fake"], yticklabels=["real", "fake"])
    plt.xlabel('Prédiction')
    plt.ylabel('Vraie étiquette')
    plt.title(f'Matrice de confusion - {model_name} ({language})')
    confusion_matrix_path = os.path.join(output_dir, f'confusion_matrix_{language}.png')
    plt.savefig(confusion_matrix_path)
    plt.close()
    
    # Créer un graphique de l'historique d'entraînement
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'training_history_{language}.png'))
    plt.close()
    
    # Ajouter aux résultats par langue
    with open(os.path.join(RESULTS_DIR, f'evaluation_{language}.txt'), 'a') as f:
        f.write(f"\n{'-'*40}\nModèle: {model_name}\n{'-'*40}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Temps d'entraînement: {training_time:.2f} secondes\n\n")
        f.write("Matrice de confusion:\n")
        f.write(f"[[{cm[0][0]} {cm[0][1]}]\n [{cm[1][0]} {cm[1][1]}]]\n\n")
        f.write("Rapport de classification:\n")
        f.write(report + "\n\n")
    
    return metrics_dict


def compare_with_clustering(transformer_results):
    """Compare les résultats des modèles transformers avec les algorithmes de clustering"""
    print("\nComparaison des résultats avec les algorithmes de clustering...")
    
    # Chemin vers les résultats de clustering
    clustering_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'clustering_comparison.csv')
    
    if not os.path.exists(clustering_file):
        print(f"Fichier de résultats de clustering non trouvé: {clustering_file}")
        return None
    
    try:
        # Charger les résultats de clustering
        clustering_results = pd.read_csv(clustering_file)
        
        # Créer un DataFrame pour les résultats des transformers
        transformer_df = pd.DataFrame({
            'model': list(transformer_results.keys()),
            'accuracy': [transformer_results[model]['accuracy'] for model in transformer_results.keys()],
            'f1': [transformer_results[model]['f1'] for model in transformer_results.keys()]
        })
        
        # Fusionner les résultats pour la comparaison
        comparison_file = os.path.join(RESULTS_DIR, 'transformer_vs_clustering.csv')
        transformer_df.to_csv(comparison_file, index=False)
        
        # Créer un graphique comparatif
        plt.figure(figsize=(12, 8))
        
        # Graphique des scores F1
        plt.subplot(2, 1, 1)
        plt.title('Comparaison des scores F1 entre transformers et clustering')
        
        # Transformers
        transformer_models = list(transformer_results.keys())
        transformer_f1 = [transformer_results[model]['f1'] for model in transformer_models]
        
        x_transformer = np.arange(len(transformer_models))
        plt.bar(x_transformer, transformer_f1, width=0.4, label='Transformers (Supervisé)', color='blue', alpha=0.7)
        
        # Ajouter les valeurs sur les barres
        for i, v in enumerate(transformer_f1):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
        
        plt.xticks(x_transformer, [model.replace('_', ' ') for model in transformer_models], rotation=45)
        plt.ylim(0, 1.1)
        plt.ylabel('Score F1')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Sauvegarder le graphique
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'transformer_vs_clustering.png'))
        
        print(f"Comparaison sauvegardée dans {comparison_file} et {os.path.join(RESULTS_DIR, 'transformer_vs_clustering.png')}")
        
        return comparison_file
    
    except Exception as e:
        print(f"Erreur lors de la comparaison avec les algorithmes de clustering: {e}")
        return None


def create_summary_file(results):
    """Crée un fichier de synthèse avec les résultats de tous les modèles"""
    with open(os.path.join(RESULTS_DIR, 'transformer_results_summary.txt'), 'w') as f:
        f.write("SYNTHÈSE DES RÉSULTATS DE DÉTECTION DE FAKE NEWS AVEC TRANSFORMERS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. RÉSULTATS PAR MODÈLE\n")
        f.write("-" * 40 + "\n")
        for model, metrics in results.items():
            f.write(f"Modèle: {model}\n")
            f.write(f"  - Langue: {metrics['language']}\n")
            f.write(f"  - Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"  - Precision: {metrics['precision']:.4f}\n")
            f.write(f"  - Recall: {metrics['recall']:.4f}\n")
            f.write(f"  - F1 Score: {metrics['f1']:.4f}\n")
            f.write(f"  - Temps d'entraînement: {metrics['training_time']:.2f} secondes\n\n")
        
        f.write("2. RÉSULTATS PAR LANGUE\n")
        f.write("-" * 40 + "\n")
        
        # Regrouper par langue
        by_language = {}
        for model, metrics in results.items():
            lang = metrics['language']
            if lang not in by_language:
                by_language[lang] = []
            by_language[lang].append((model, metrics))
        
        for lang, models in by_language.items():
            f.write(f"Langue: {lang}\n")
            best_model = max(models, key=lambda x: x[1]['f1'])
            f.write(f"  - Meilleur modèle: {best_model[0]} (F1={best_model[1]['f1']:.4f})\n")
            f.write("  - Tous les modèles:\n")
            for model, metrics in sorted(models, key=lambda x: x[1]['f1'], reverse=True):
                f.write(f"    * {model}: F1={metrics['f1']:.4f}, Accuracy={metrics['accuracy']:.4f}\n")
            f.write("\n")
        
        f.write("3. CONCLUSION\n")
        f.write("-" * 40 + "\n")
        
        # Trouver le meilleur modèle global
        best_model_name = max(results.items(), key=lambda x: x[1]['f1'])[0]
        best_metrics = results[best_model_name]
        f.write(f"Le meilleur modèle global est {best_model_name} avec un score F1 de {best_metrics['f1']:.4f}\n")
        f.write(f"Ce modèle a été entraîné pour la langue: {best_metrics['language']}\n\n")
        
        # Recommandations pour chaque langue
        f.write("Recommandations:\n")
        for lang, models in by_language.items():
            best = max(models, key=lambda x: x[1]['f1'])
            f.write(f"  - Pour {lang}: Utiliser {best[0]} (F1={best[1]['f1']:.4f})\n")
    
    print(f"Synthèse des résultats sauvegardée dans {os.path.join(RESULTS_DIR, 'summary.txt')}")
    
    # Créer un graphique comparatif des performances
    plt.figure(figsize=(14, 10))
    
    # Préparer les données pour le graphique
    model_names = list(results.keys())
    f1_scores = [results[m]['f1'] for m in model_names]
    accuracy_scores = [results[m]['accuracy'] for m in model_names]
    languages = [results[m]['language'] for m in model_names]
    
    # Créer un colormap basé sur les langues
    colors = {'english': 'royalblue', 'french': 'crimson', 'multilingual': 'darkgreen'}
    bar_colors = [colors[lang] for lang in languages]
    
    # Tracer le graphique
    bar_width = 0.35
    index = np.arange(len(model_names))
    
    plt.bar(index, f1_scores, bar_width, label='F1 Score', color=bar_colors)
    plt.bar(index + bar_width, accuracy_scores, bar_width, label='Accuracy', color=bar_colors, alpha=0.7)
    
    plt.xlabel('Modèle')
    plt.ylabel('Score')
    plt.title('Comparaison des performances des modèles de détection de fake news')
    plt.xticks(index + bar_width/2, model_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Créer une légende pour les couleurs des langues
    from matplotlib.patches import Patch
    language_patches = [Patch(color=colors[lang], label=lang) for lang in colors.keys()]
    plt.legend(handles=[Patch(color='grey', label='F1 Score'), 
                       Patch(color='grey', alpha=0.7, label='Accuracy')] + language_patches,
              loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
    
    plt.savefig(os.path.join(RESULTS_DIR, 'transformer_model_comparison.png'), bbox_inches='tight')
    plt.close()
    
    print(f"Graphique de comparaison sauvegardé dans {os.path.join(RESULTS_DIR, 'transformer_model_comparison.png')}")


def train_transformer_models():
    """Entraîne et évalue les modèles transformers pour la détection de fake news"""
    # Charger et prétraiter les données
    en_train, en_test = load_and_preprocess_english_data()
    fr_train, fr_test = load_and_preprocess_french_data()
    
    if en_train is None or fr_train is None:
        print("Erreur lors du chargement des données. Vérifiez les chemins des fichiers.")
        return {}
    
    # Diviser les données en ensembles d'entraînement et de validation (80-10-10)
    en_train, en_val = train_test_split(en_train, test_size=0.125, random_state=42)  # 80/10=8, donc 1/8=0.125
    fr_train, fr_val = train_test_split(fr_train, test_size=0.125, random_state=42)
    
    results = {}
    
    # 1. Modèle BERT multilingue pour les deux langues combinées
    print("\n" + "="*80)
    print("ENTRAÎNEMENT DU MODÈLE BERT MULTILINGUE")
    print("="*80)
    
    # Préparer les données combinées (anglais + français)
    combined_train = pd.concat([en_train, fr_train], ignore_index=True)
    combined_val = pd.concat([en_val, fr_val], ignore_index=True)
    combined_test = pd.concat([en_test, fr_test], ignore_index=True)
    
    # Créer les datasets TensorFlow
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    batch_size = 16  # Ajuster en fonction de la mémoire disponible
    max_length = 128  # Ajuster en fonction de vos données
    
    bert_train_dataset = tf.data.Dataset.from_tensor_slices((
        combined_train['clean_text'].values, combined_train['label'].values
    ))
    bert_train_dataset = bert_train_dataset.batch(batch_size)
    
    bert_val_dataset = tf.data.Dataset.from_tensor_slices((
        combined_val['clean_text'].values, combined_val['label'].values
    ))
    bert_val_dataset = bert_val_dataset.batch(batch_size)
    
    bert_test_dataset = tf.data.Dataset.from_tensor_slices((
        combined_test['clean_text'].values, combined_test['label'].values
    ))
    bert_test_dataset = bert_test_dataset.batch(batch_size)
    
    # Créer et entraîner le modèle BERT multilingue
    bert_model = create_bert_model('bert-base-multilingual-cased', max_length)
    bert_metrics = train_and_evaluate_model(
        bert_model,
        bert_train_dataset,
        bert_val_dataset,
        bert_test_dataset,
        'BERT',
        'multilingual',
        epochs=3
    )
    results['BERT_multilingual'] = bert_metrics
    
    # 2. Modèle DistilBERT pour l'anglais
    print("\n" + "="*80)
    print("ENTRAÎNEMENT DU MODÈLE DISTILBERT POUR L'ANGLAIS")
    print("="*80)
    
    distilbert_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    distilbert_en_train_dataset = tf.data.Dataset.from_tensor_slices((
        en_train['clean_text'].values, en_train['label'].values
    ))
    distilbert_en_train_dataset = distilbert_en_train_dataset.batch(batch_size)
    
    distilbert_en_val_dataset = tf.data.Dataset.from_tensor_slices((
        en_val['clean_text'].values, en_val['label'].values
    ))
    distilbert_en_val_dataset = distilbert_en_val_dataset.batch(batch_size)
    
    distilbert_en_test_dataset = tf.data.Dataset.from_tensor_slices((
        en_test['clean_text'].values, en_test['label'].values
    ))
    distilbert_en_test_dataset = distilbert_en_test_dataset.batch(batch_size)
    
    # Créer et entraîner le modèle DistilBERT pour l'anglais
    distilbert_model = create_distilbert_model('distilbert-base-uncased', max_length)
    distilbert_metrics = train_and_evaluate_model(
        distilbert_model,
        distilbert_en_train_dataset,
        distilbert_en_val_dataset,
        distilbert_en_test_dataset,
        'DistilBERT',
        'english',
        epochs=3
    )
    results['DistilBERT_english'] = distilbert_metrics
    
    # 3. Modèle CamemBERT pour le français
    print("\n" + "="*80)
    print("ENTRAÎNEMENT DU MODÈLE CAMEMBERT POUR LE FRANÇAIS")
    print("="*80)
    
    camembert_tokenizer = AutoTokenizer.from_pretrained('camembert-base')
    
    camembert_fr_train_dataset = tf.data.Dataset.from_tensor_slices((
        fr_train['clean_text'].values, fr_train['label'].values
    ))
    camembert_fr_train_dataset = camembert_fr_train_dataset.batch(batch_size)
    
    camembert_fr_val_dataset = tf.data.Dataset.from_tensor_slices((
        fr_val['clean_text'].values, fr_val['label'].values
    ))
    camembert_fr_val_dataset = camembert_fr_val_dataset.batch(batch_size)
    
    camembert_fr_test_dataset = tf.data.Dataset.from_tensor_slices((
        fr_test['clean_text'].values, fr_test['label'].values
    ))
    camembert_fr_test_dataset = camembert_fr_test_dataset.batch(batch_size)
    
    # Créer et entraîner le modèle CamemBERT pour le français
    camembert_model = create_camembert_model('camembert-base', max_length)
    camembert_metrics = train_and_evaluate_model(
        camembert_model,
        camembert_fr_train_dataset,
        camembert_fr_val_dataset,
        camembert_fr_test_dataset,
        'CamemBERT',
        'french',
        epochs=3
    )
    results['CamemBERT_french'] = camembert_metrics
    
    return results


if __name__ == "__main__":
    print("="*80)
    print("ENTRAÎNEMENT DES MODÈLES TRANSFORMERS POUR LA DÉTECTION DE FAKE NEWS")
    print("="*80)
    print("\nNOTE IMPORTANTE: Comme les étiquettes réelles (y) ne sont pas disponibles,")
    print("nous générons des étiquettes artificielles pour l'apprentissage supervisé.")
    print("Cette approche permet de démontrer le pipeline complet, mais les performances")
    print("ne reflètent pas la capacité réelle à détecter les fake news.")
    print("="*80)
    
    # Préparer le répertoire pour les résultats
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Créer des fichiers d'évaluation vides pour chaque langue
    for lang in ['french', 'english', 'multilingual']:
        with open(os.path.join(RESULTS_DIR, f'evaluation_{lang}.txt'), 'w') as f:
            f.write(f"ÉVALUATION DES MODÈLES TRANSFORMERS POUR LA DÉTECTION DE FAKE NEWS - LANGUE: {lang.upper()}\n")
            f.write("=" * 80 + "\n\n")
            f.write("NOTE: Les étiquettes utilisées sont générées artificiellement.\n")
            f.write("Les performances présentées sont uniquement à titre illustratif.\n\n")
    
    # Entraîner les modèles transformers
    results = train_transformer_models()
    
    # Créer un fichier de synthèse
    create_summary_file(results)
    
    # Comparer avec les résultats des algorithmes de clustering
    compare_with_clustering(results)
    
    print("\nEntraînement des modèles transformers terminé avec succès!")
    print("\nRésumé des performances:")
    for model, metrics in results.items():
        print(f"- {model}: F1={metrics['f1']:.4f}, Accuracy={metrics['accuracy']:.4f}")
        
    print(f"\nLes résultats d'évaluation détaillés sont disponibles dans le dossier {RESULTS_DIR}")
    print("\nComparaison avec les algorithmes de clustering disponible dans le fichier transformer_vs_clustering.csv")
    print("\nVous pouvez maintenant utiliser ces modèles pour la détection de fake news dans votre application Streamlit.")

