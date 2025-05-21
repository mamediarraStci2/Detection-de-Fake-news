import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from langdetect import detect
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch
import os

# Télécharger les ressources NLTK nécessaires
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Charger les modèles spaCy pour le français et l'anglais
try:
    nlp_fr = spacy.load('fr_core_news_sm')
    nlp_en = spacy.load('en_core_web_sm')
except OSError:
    print("Téléchargement des modèles spaCy...")
    os.system("python -m spacy download fr_core_news_sm")
    os.system("python -m spacy download en_core_web_sm")
    nlp_fr = spacy.load('fr_core_news_sm')
    nlp_en = spacy.load('en_core_web_sm')

# Initialiser le lemmatizer de NLTK
lemmatizer = WordNetLemmatizer()

# Stopwords pour le français et l'anglais
stop_words_fr = set(stopwords.words('french'))
stop_words_en = set(stopwords.words('english'))

def detect_language(text):
    """Détecte la langue du texte (français ou anglais)"""
    try:
        return detect(text)
    except:
        return 'en'  # Par défaut, on considère que c'est de l'anglais

def clean_text(text, language='en'):
    """Nettoie le texte en fonction de la langue détectée"""
    if not isinstance(text, str):
        return ""
    
    # Convertir en minuscules
    text = text.lower()
    
    # Supprimer les URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Supprimer les caractères spéciaux et les chiffres
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Tokenisation et suppression des stopwords en fonction de la langue
    if language.startswith('fr'):
        # Utiliser spaCy pour le français
        doc = nlp_fr(text)
        tokens = [token.lemma_ for token in doc if token.text.strip() and not token.is_stop]
        return ' '.join(tokens)
    else:
        # Utiliser NLTK pour l'anglais
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words_en]
        return ' '.join(tokens)

def preprocess_data(df, text_column, is_french=False):
    """Prétraite les données du DataFrame"""
    # Copier le DataFrame pour éviter de modifier l'original
    df_processed = df.copy()
    
    # Détecter la langue si non spécifiée
    if not is_french:
        # Échantillonner quelques textes pour déterminer la langue dominante
        sample_texts = df[text_column].dropna().sample(min(100, len(df))).tolist()
        languages = [detect_language(text) for text in sample_texts if isinstance(text, str)]
        dominant_language = 'fr' if languages.count('fr') > languages.count('en') else 'en'
    else:
        dominant_language = 'fr'
    
    # Nettoyer les textes
    df_processed['clean_text'] = df_processed[text_column].apply(
        lambda x: clean_text(x, language=dominant_language)
    )
    
    return df_processed

def get_tfidf_features(train_data, test_data=None, max_features=5000):
    """Extrait les caractéristiques TF-IDF des textes"""
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(train_data)
    
    if test_data is not None:
        X_test = vectorizer.transform(test_data)
        return X_train, X_test, vectorizer
    
    return X_train, vectorizer

def get_bert_embeddings(texts, model_name='bert-base-multilingual-cased'):
    """Extrait les embeddings BERT des textes"""
    # Charger le tokenizer et le modèle BERT multilingue
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Mettre le modèle en mode évaluation
    model.eval()
    
    embeddings = []
    batch_size = 8  # Ajuster en fonction de la mémoire disponible
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokeniser les textes
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, 
                                 max_length=128, return_tensors='pt')
        
        # Obtenir les embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
            
        # Utiliser la moyenne des dernières couches cachées comme embedding
        batch_embeddings = model_output.last_hidden_state.mean(dim=1)
        embeddings.append(batch_embeddings)
    
    # Concaténer tous les embeddings
    all_embeddings = torch.cat(embeddings, dim=0).numpy()
    
    return all_embeddings

def load_and_preprocess_data(english_path, french_path):
    """Charge et prétraite les données anglaises et françaises"""
    # Charger les données
    print("Chargement des données...")
    
    # Données anglaises
    df_en_train = pd.read_csv(english_path + '/train.csv', delimiter=';')
    df_en_test = pd.read_csv(english_path + '/test (1).csv', delimiter=';')
    
    # Données françaises
    df_fr_train = pd.read_csv(french_path + '/train.csv', delimiter=';')
    df_fr_test = pd.read_csv(french_path + '/test.csv', delimiter=';')
    
    # Identifier les colonnes de texte
    en_text_col = 'text' if 'text' in df_en_train.columns else df_en_train.columns[2]
    fr_text_col = 'post' if 'post' in df_fr_train.columns else df_fr_train.columns[1]
    
    # Prétraiter les données
    print("Prétraitement des données anglaises...")
    df_en_train_processed = preprocess_data(df_en_train, en_text_col, is_french=False)
    df_en_test_processed = preprocess_data(df_en_test, en_text_col, is_french=False)
    
    print("Prétraitement des données françaises...")
    df_fr_train_processed = preprocess_data(df_fr_train, fr_text_col, is_french=True)
    df_fr_test_processed = preprocess_data(df_fr_test, fr_text_col, is_french=True)
    
    return {
        'english': {
            'train': df_en_train_processed,
            'test': df_en_test_processed,
            'text_col': en_text_col
        },
        'french': {
            'train': df_fr_train_processed,
            'test': df_fr_test_processed,
            'text_col': fr_text_col
        }
    }
