"""
Analyse de base des datasets de fake news en français et en anglais
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter
from langdetect import detect
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

# Téléchargement des ressources NLTK nécessaires
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Chemins des datasets
ENGLISH_TRAIN_PATH = "Fake_news/train.csv"
ENGLISH_TEST_PATH = "Fake_news/test (1).csv"
FRENCH_TRAIN_PATH = "French_Fake_News/train.csv"
FRENCH_TEST_PATH = "French_Fake_News/test.csv"

# Création du répertoire pour les résultats
os.makedirs("analysis_results", exist_ok=True)

def clean_text(text):
    """Nettoie le texte en supprimant la ponctuation, les chiffres, etc."""
    if not isinstance(text, str):
        return ""
    
    # Convertir en minuscules
    text = text.lower()
    
    # Supprimer les URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Supprimer les caractères spéciaux et les chiffres
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    return text

def remove_stopwords(text, language='en'):
    """Supprime les mots vides (stopwords) selon la langue"""
    if language == 'fr':
        stop_words = set(stopwords.words('french'))
        tokens = word_tokenize(text, language='french')
    else:
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        
    return ' '.join([word for word in tokens if word not in stop_words])

def detect_language(text):
    """Détecte la langue du texte (français ou anglais)"""
    try:
        return detect(text)
    except:
        return 'en'  # Par défaut, on considère l'anglais

def load_datasets():
    """Charge les datasets français et anglais"""
    print("Chargement des datasets...")
    
    # Chargement des données anglaises
    english_train = pd.read_csv(ENGLISH_TRAIN_PATH, delimiter=';')
    english_test = pd.read_csv(ENGLISH_TEST_PATH, delimiter=';')
    
    # Chargement des données françaises
    french_train = pd.read_csv(FRENCH_TRAIN_PATH, delimiter=';')
    french_test = pd.read_csv(FRENCH_TEST_PATH, delimiter=';')
    
    return {
        'english': {
            'train': english_train,
            'test': english_test
        },
        'french': {
            'train': french_train,
            'test': french_test
        }
    }

def analyze_dataset_structure(datasets):
    """Analyse la structure des datasets"""
    print("\nAnalyse de la structure des datasets:")
    
    for lang, data in datasets.items():
        for split, df in data.items():
            print(f"\n{lang.capitalize()} {split} dataset:")
            print(f"  - Nombre d'échantillons: {len(df)}")
            print(f"  - Colonnes: {', '.join(df.columns)}")
            print(f"  - Types de données:")
            for col in df.columns:
                print(f"    - {col}: {df[col].dtype}")
            
            # Vérification des valeurs manquantes
            missing_values = df.isnull().sum()
            print(f"  - Valeurs manquantes:")
            for col, count in missing_values.items():
                if count > 0:
                    print(f"    - {col}: {count} ({count/len(df)*100:.2f}%)")

def analyze_text_length(datasets):
    """Analyse la longueur des textes dans les datasets"""
    print("\nAnalyse de la longueur des textes:")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for i, (lang, data) in enumerate(datasets.items()):
        for j, (split, df) in enumerate(data.items()):
            # Détermination de la colonne de texte
            text_col = 'text' if 'text' in df.columns else 'post'
            
            # Calcul de la longueur des textes
            if text_col in df.columns:
                df['text_length'] = df[text_col].fillna('').apply(lambda x: len(str(x).split()))
                
                # Tracé de la distribution
                ax = axes[i, j]
                sns.histplot(df['text_length'], bins=50, kde=True, ax=ax)
                ax.set_title(f"{lang.capitalize()} {split} - Distribution de la longueur des textes")
                ax.set_xlabel("Nombre de mots")
                ax.set_ylabel("Fréquence")
                
                # Statistiques
                print(f"\n{lang.capitalize()} {split}:")
                print(f"  - Longueur moyenne: {df['text_length'].mean():.2f} mots")
                print(f"  - Longueur médiane: {df['text_length'].median():.2f} mots")
                print(f"  - Longueur minimale: {df['text_length'].min()} mots")
                print(f"  - Longueur maximale: {df['text_length'].max()} mots")
    
    plt.tight_layout()
    plt.savefig("analysis_results/text_length_distribution.png")
    plt.close()

def generate_word_clouds(datasets):
    """Génère des nuages de mots pour chaque dataset"""
    print("\nGénération des nuages de mots:")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for i, (lang, data) in enumerate(datasets.items()):
        for j, (split, df) in enumerate(data.items()):
            # Détermination de la colonne de texte
            text_col = 'text' if 'text' in df.columns else 'post'
            
            # Préparation du texte
            if text_col in df.columns:
                # Concaténation de tous les textes
                all_text = ' '.join(df[text_col].fillna('').astype(str))
                
                # Nettoyage du texte
                all_text = clean_text(all_text)
                
                # Suppression des stopwords
                language = 'fr' if lang == 'french' else 'en'
                all_text = remove_stopwords(all_text, language=language)
                
                # Génération du nuage de mots
                wordcloud = WordCloud(
                    width=800, height=600,
                    background_color='white',
                    max_words=200,
                    contour_width=3,
                    contour_color='steelblue'
                ).generate(all_text)
                
                # Affichage du nuage de mots
                ax = axes[i, j]
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.set_title(f"{lang.capitalize()} {split} - Nuage de mots")
                ax.axis('off')
                
                # Sauvegarde du nuage de mots
                wordcloud.to_file(f"analysis_results/wordcloud_{lang}_{split}.png")
    
    plt.tight_layout()
    plt.savefig("analysis_results/wordclouds.png")
    plt.close()

def analyze_common_words(datasets):
    """Analyse les mots les plus fréquents dans chaque dataset"""
    print("\nAnalyse des mots les plus fréquents:")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for i, (lang, data) in enumerate(datasets.items()):
        for j, (split, df) in enumerate(data.items()):
            # Détermination de la colonne de texte
            text_col = 'text' if 'text' in df.columns else 'post'
            
            # Préparation du texte
            if text_col in df.columns:
                # Concaténation de tous les textes
                all_text = ' '.join(df[text_col].fillna('').astype(str))
                
                # Nettoyage du texte
                all_text = clean_text(all_text)
                
                # Suppression des stopwords
                language = 'fr' if lang == 'french' else 'en'
                all_text = remove_stopwords(all_text, language=language)
                
                # Comptage des mots
                words = all_text.split()
                word_counts = Counter(words)
                
                # Extraction des 20 mots les plus fréquents
                most_common = word_counts.most_common(20)
                
                # Tracé du graphique
                ax = axes[i, j]
                words, counts = zip(*most_common)
                sns.barplot(x=list(counts), y=list(words), ax=ax)
                ax.set_title(f"{lang.capitalize()} {split} - Mots les plus fréquents")
                ax.set_xlabel("Fréquence")
                ax.set_ylabel("Mot")
                
                # Affichage des mots les plus fréquents
                print(f"\n{lang.capitalize()} {split} - Mots les plus fréquents:")
                for word, count in most_common:
                    print(f"  - {word}: {count}")
    
    plt.tight_layout()
    plt.savefig("analysis_results/common_words.png")
    plt.close()

def main():
    """Fonction principale"""
    # Chargement des datasets
    datasets = load_datasets()
    
    # Analyse de la structure des datasets
    analyze_dataset_structure(datasets)
    
    # Analyse de la longueur des textes
    analyze_text_length(datasets)
    
    # Génération des nuages de mots
    generate_word_clouds(datasets)
    
    # Analyse des mots les plus fréquents
    analyze_common_words(datasets)
    
    print("\nAnalyse de base terminée. Les résultats sont disponibles dans le répertoire 'analysis_results'.")

if __name__ == "__main__":
    main()
