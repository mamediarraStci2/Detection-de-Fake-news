"""
Module de prétraitement des données pour la détection de fake news
"""
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
import joblib
import os

# Téléchargement des ressources NLTK nécessaires
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Chargement des modèles spaCy
try:
    nlp_en = spacy.load('en_core_web_sm')
    nlp_fr = spacy.load('fr_core_news_sm')
except OSError:
    print("Téléchargement des modèles spaCy...")
    os.system('python -m spacy download en_core_web_sm')
    os.system('python -m spacy download fr_core_news_sm')
    nlp_en = spacy.load('en_core_web_sm')
    nlp_fr = spacy.load('fr_core_news_sm')

class TextPreprocessor:
    """
    Classe pour le prétraitement des textes en français et en anglais
    """
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words_en = set(stopwords.words('english'))
        self.stop_words_fr = set(stopwords.words('french'))
        self.tokenizer = None
        self.model = None
        
    def detect_language(self, text):
        """Détecte la langue du texte (français ou anglais)"""
        try:
            return detect(text)
        except:
            return 'en'  # Par défaut, on considère l'anglais
    
    def clean_text(self, text, language='en'):
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
    
    def remove_stopwords(self, text, language='en'):
        """Supprime les mots vides (stopwords) selon la langue"""
        if language == 'fr':
            stop_words = self.stop_words_fr
            tokens = word_tokenize(text, language='french')
        else:
            stop_words = self.stop_words_en
            tokens = word_tokenize(text)
            
        return ' '.join([word for word in tokens if word not in stop_words])
    
    def lemmatize_text(self, text, language='en'):
        """Lemmatise le texte selon la langue"""
        if language == 'fr':
            doc = nlp_fr(text)
            return ' '.join([token.lemma_ for token in doc if token.lemma_ != '-PRON-'])
        else:
            doc = nlp_en(text)
            return ' '.join([token.lemma_ for token in doc if token.lemma_ != '-PRON-'])
    
    def preprocess_text(self, text):
        """Applique toutes les étapes de prétraitement au texte"""
        if not isinstance(text, str) or text.strip() == "":
            return ""
        
        # Détection de la langue
        language = self.detect_language(text)
        if language not in ['en', 'fr']:
            language = 'en'  # Par défaut, on considère l'anglais
        
        # Nettoyage du texte
        text = self.clean_text(text, language)
        
        # Suppression des stopwords
        text = self.remove_stopwords(text, language)
        
        # Lemmatisation
        text = self.lemmatize_text(text, language)
        
        return text
    
    def load_bert_model(self):
        """Charge le modèle BERT multilingue pour les embeddings"""
        if self.tokenizer is None or self.model is None:
            print("Chargement du modèle BERT multilingue...")
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
            self.model = AutoModel.from_pretrained('bert-base-multilingual-cased')
    
    def get_bert_embedding(self, text):
        """Obtient l'embedding BERT d'un texte"""
        self.load_bert_model()
        
        # Tokenisation et préparation pour BERT
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # Obtention des embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Utilisation de la moyenne des dernières couches cachées comme embedding
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        
        return embeddings
    
    def preprocess_dataframe(self, df, text_column='text', title_column='title'):
        """Prétraite les textes dans un DataFrame"""
        # Combinaison du titre et du texte si les deux sont disponibles
        if title_column in df.columns and text_column in df.columns:
            df['combined_text'] = df[title_column].fillna('') + ' ' + df[text_column].fillna('')
        elif text_column in df.columns:
            df['combined_text'] = df[text_column].fillna('')
        elif title_column in df.columns:
            df['combined_text'] = df[title_column].fillna('')
        else:
            raise ValueError(f"Les colonnes {text_column} et {title_column} n'existent pas dans le DataFrame")
        
        # Prétraitement des textes
        print("Prétraitement des textes...")
        df['processed_text'] = df['combined_text'].apply(self.preprocess_text)
        
        return df
    
    def get_tfidf_vectors(self, texts, max_features=5000):
        """Obtient les vecteurs TF-IDF des textes"""
        vectorizer = TfidfVectorizer(max_features=max_features)
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        return tfidf_matrix, vectorizer
    
    def get_bert_embeddings_batch(self, texts, batch_size=8):
        """Obtient les embeddings BERT pour un lot de textes"""
        self.load_bert_model()
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenisation et préparation pour BERT
            inputs = self.tokenizer(batch_texts, return_tensors="pt", truncation=True, 
                                    padding=True, max_length=512)
            
            # Obtention des embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Utilisation de la moyenne des dernières couches cachées comme embedding
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
            all_embeddings.append(batch_embeddings)
        
        # Concaténation des embeddings de tous les lots
        return np.vstack(all_embeddings)

def load_and_preprocess_data(english_train_path, english_test_path, french_train_path, french_test_path):
    """
    Charge et prétraite les données anglaises et françaises
    """
    # Chargement des données
    print("Chargement des données anglaises...")
    english_train = pd.read_csv(english_train_path, delimiter=';')
    english_test = pd.read_csv(english_test_path, delimiter=';')
    
    print("Chargement des données françaises...")
    french_train = pd.read_csv(french_train_path, delimiter=';')
    french_test = pd.read_csv(french_test_path, delimiter=';')
    
    # Prétraitement des données
    preprocessor = TextPreprocessor()
    
    print("Prétraitement des données anglaises...")
    english_train = preprocessor.preprocess_dataframe(english_train)
    english_test = preprocessor.preprocess_dataframe(english_test)
    
    print("Prétraitement des données françaises...")
    french_train = preprocessor.preprocess_dataframe(french_train, text_column='post')
    french_test = preprocessor.preprocess_dataframe(french_test, text_column='post')
    
    # Ajout d'une colonne pour la langue
    english_train['language'] = 'en'
    english_test['language'] = 'en'
    french_train['language'] = 'fr'
    french_test['language'] = 'fr'
    
    # Fusion des données
    train_data = pd.concat([english_train, french_train], ignore_index=True)
    test_data = pd.concat([english_test, french_test], ignore_index=True)
    
    return train_data, test_data, preprocessor

if __name__ == "__main__":
    # Chemins des données
    english_train_path = "Fake_news/train.csv"
    english_test_path = "Fake_news/test (1).csv"
    french_train_path = "French_Fake_News/train.csv"
    french_test_path = "French_Fake_News/test.csv"
    
    # Chargement et prétraitement des données
    train_data, test_data, preprocessor = load_and_preprocess_data(
        english_train_path, english_test_path, french_train_path, french_test_path
    )
    
    # Sauvegarde des données prétraitées
    print("Sauvegarde des données prétraitées...")
    train_data.to_csv("preprocessed_train_data.csv", index=False)
    test_data.to_csv("preprocessed_test_data.csv", index=False)
    
    print("Prétraitement terminé !")
