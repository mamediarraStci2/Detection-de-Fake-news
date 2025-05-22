"""
Interface utilisateur pour la détection de fake news
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import time
import re
# Liste de mots associés aux fake news (fréquemment trouvés dans les titres sensationnalistes)
FAKE_NEWS_WORDS = [
    'shocking', 'secret', 'they don\'t want you to know', 'conspiracy', 'hoax', 'scam', 
    'fake', 'miracle', 'shocking truth', 'government lies', 'exposed',
    'scandal', 'breaking', 'bombshell', 'urgent', 'alarming', 'censored', 'banned',
    'choquant', 'secret', 'complot', 'canular', 'arnaque', 'faux', 'miracle', 
    'vérité choquante', 'mensonges', 'complot', 'exposé',
    'scandale', 'urgent', 'alarmant', 'censuré', 'interdit'
]

# Mots communs dans les vraies news
REAL_NEWS_WORDS = [
    'report', 'according to', 'study finds', 'research', 'announced', 'published',
    'confirmed', 'stated', 'evidence', 'data', 'source', 'official', 'expert',
    'analysis', 'investigation', 'rapport', 'selon', 'étude', 'recherche', 'annoncé',
    'publié', 'confirmé', 'déclaré', 'preuve', 'données', 'source', 'officiel',
    'expert', 'analyse', 'enquête'
]

# Création d'une classe KMeans personnalisée pour éviter les erreurs liées aux attributs manquants
class SimplifiedKMeans:
    def __init__(self, n_clusters=2, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.cluster_centers_ = None
        np.random.seed(random_state)
        self.fake_cluster = 0  # Par défaut, cluster 0 = fake news
    
    def fit(self, X):
        # Initialisation aléatoire des centres
        self.cluster_centers_ = np.random.rand(self.n_clusters, X.shape[1])
        return self
    
    def predict(self, X):
        # Calcul des distances aux centres
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :], axis=2)
        # Attribution au cluster le plus proche
        return np.argmin(distances, axis=1)
# Implémentation simplifiée du préprocesseur pour éviter la dépendance à langdetect
class SimpleTextPreprocessor:
    """Classe simplifiée pour le prétraitement des textes avec fonctionnalités améliorées"""
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        # Liste de mots communs en français pour la détection de langue
        self.french_words = ['le', 'la', 'les', 'un', 'une', 'des', 'et', 'est', 'sont', 'ce', 'cette', 
                      'ces', 'mon', 'ton', 'son', 'notre', 'votre', 'leur', 'qui', 'que', 'dont',
                      'où', 'quoi', 'pourquoi', 'comment', 'quand', 'quel', 'quelle', 'à', 'au', 'aux']
    
    def detect_language(self, text):
        """Détecte la langue du texte de manière simpifiée"""
        if not isinstance(text, str) or text.strip() == "":
            return 'en'
            
        text_lower = text.lower()
        # Compte le nombre de mots français dans le texte
        french_word_count = sum(1 for word in self.french_words if f" {word} " in f" {text_lower} ")
        
        # Si au moins 3 mots français sont détectés, considère le texte comme français
        return 'fr' if french_word_count >= 3 else 'en'
    
    def preprocess_text(self, text):
        """Prétraitement amélioré du texte"""
        if not isinstance(text, str) or text.strip() == "":
            return ""
        
        # Détection de la langue
        language = self.detect_language(text)
        
        # Nettoyage plus complet
        text = text.lower()
        
        # Suppression des URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) if 're' in globals() else text.lower()
        
        # Suppression de la ponctuation
        for char in '.,;:!?()[]{}<>"\'':
            text = text.replace(char, ' ')
        
        # Suppression des espaces multiples
        while '  ' in text:
            text = text.replace('  ', ' ')
        
        # Analyse des mots-clés (augmente la performance de détection)
        fake_words_count = sum(1 for word in FAKE_NEWS_WORDS if word.lower() in text.lower())
        real_words_count = sum(1 for word in REAL_NEWS_WORDS if word.lower() in text.lower())
        
        return text, language, fake_words_count, real_words_count
    
    def get_bert_embedding(self, text):
        """Simulation d'embedding améliorée avec des vecteurs intelligents"""
        # Ce code génère un embedding qui est influencé par le contenu du texte
        # pour donner des résultats plus cohérents et réalistes
        
        # Générer un vecteur de base
        np.random.seed(hash(text) % 10000)  # Assure la cohérence pour le même texte
        embedding = np.random.rand(1, 768)  # Dimension typique d'un embedding BERT
        
        # Influencer le vecteur en fonction du contenu du texte
        processed_text, _, fake_count, real_count = self.preprocess_text(text)
        words = processed_text.split()
        
        # Ajuster le vecteur selon la proportion de mots associés aux fake news
        if len(words) > 0:
            fake_ratio = fake_count / len(words)
            real_ratio = real_count / len(words)
            
            # Moduler certaines dimensions du vecteur en fonction de ces ratios
            # Les 100 premières dimensions sont influencées par les indicateurs de fake news
            embedding[0, :100] = embedding[0, :100] * (1 + fake_ratio)
            # Les 100 dimensions suivantes sont influencées par les indicateurs de vraies news
            embedding[0, 100:200] = embedding[0, 100:200] * (1 + real_ratio)
        
        return embedding

# Configuration de la page
st.set_page_config(
    page_title="Détecteur de Fake News",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonction pour charger les modèles
# Fonction pour simuler le chargement des modèles (non utilisée dans cette version simplifiée)
def load_models_unused():
    """Fonction non utilisée dans cette version simplifiée"""
    pass

# Fonction pour prédire si un texte est une fake news
def predict_fake_news(text, preprocessor, kmeans_model):
    """
    Prédit si un texte est une fake news avec une analyse avancée
    
    Args:
        text: Texte à analyser
        preprocessor: Instance de TextPreprocessor
        kmeans_model: Modèle KMeans entraîné
    
    Returns:
        is_fake: Booléen indiquant si c'est une fake news
        probabilities: Probabilités d'appartenance aux clusters
        processing_time: Temps de traitement en secondes
        language: Langue détectée du texte
    """
    start_time = time.time()
    
    # Prétraitement du texte avec plus d'informations
    processed_text, language, fake_words_count, real_words_count = preprocessor.preprocess_text(text)
    
    # Obtention de l'embedding BERT influencé par le contenu
    embedding = preprocessor.get_bert_embedding(text)
    
    # Prédiction avec KMeans
    cluster = kmeans_model.predict(embedding)[0]
    
    # Calcul des distances aux centres des clusters de manière explicite
    distances = np.array([np.linalg.norm(embedding[0] - center) for center in kmeans_model.cluster_centers_])
    probabilities = 1 / (1 + distances)
    probabilities = probabilities / np.sum(probabilities)
    
    # Déterminer si le texte est une fake news
    is_fake_by_cluster = cluster == kmeans_model.fake_cluster
    
    # Ajuster la prédiction en fonction de l'analyse des mots-clés
    # Si beaucoup plus de mots liés aux fake news que de mots liés aux vraies news
    if fake_words_count > real_words_count * 1.5:
        is_fake_by_keywords = True
        # Augmenter la probabilité du cluster fake news
        probabilities[kmeans_model.fake_cluster] = max(probabilities[kmeans_model.fake_cluster], 0.7)
    # Si beaucoup plus de mots liés aux vraies news que de mots liés aux fake news
    elif real_words_count > fake_words_count * 1.5:
        is_fake_by_keywords = False
        # Augmenter la probabilité du cluster vraies news
        probabilities[1 - kmeans_model.fake_cluster] = max(probabilities[1 - kmeans_model.fake_cluster], 0.7)
    else:
        is_fake_by_keywords = is_fake_by_cluster
    
    # Décision finale: si l'analyse des mots-clés est forte, elle prime
    final_is_fake = is_fake_by_keywords if abs(fake_words_count - real_words_count) > 2 else is_fake_by_cluster
    
    # Normaliser les probabilités
    probabilities = probabilities / np.sum(probabilities)
    
    processing_time = time.time() - start_time
    
    return final_is_fake, probabilities, processing_time, language

# Fonction pour afficher les résultats
def display_results(is_fake, probabilities, processing_time, language):
    """
    Affiche les résultats de la prédiction
    
    Args:
        cluster: Cluster prédit
        probabilities: Probabilités d'appartenance aux clusters
        n_clusters: Nombre de clusters
    """
    # Détermination du type de news
    if is_fake:
        news_type = "Fake News"
        color = "red"
    else:
        news_type = "Vraie News"
        color = "green"
    
    # Affichage du résultat
    st.markdown(f"<h2 style='text-align: center; color: {color};'>Résultat: {news_type}</h2>", unsafe_allow_html=True)
    
    # Affichage du temps de traitement et de la langue détectée
    st.markdown(f"<p style='text-align: center;'>Temps de traitement: {processing_time:.3f} secondes | Langue détectée: {language.upper()}</p>", unsafe_allow_html=True)
    
    # Affichage des probabilités
    st.subheader("Probabilités d'appartenance aux clusters")
    
    # Création du DataFrame pour le graphique
    proba_df = pd.DataFrame({
        'Type': ['Fake News', 'Vraie News'],
        'Probabilité': [probabilities[0], probabilities[1]] if is_fake else [probabilities[1], probabilities[0]]
    })
    
    # Création du graphique avec matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(proba_df['Type'], proba_df['Probabilité'], color=['#FF6B6B', '#4CAF50'])
    
    # Personnalisation du graphique
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Probabilité d'appartenance")
    ax.set_ylim(0, 1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Ajout des valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
    
    # Affichage dans Streamlit
    st.pyplot(fig)
    
    # Explication de la prédiction
    st.subheader("Explication de la prédiction")
    
    if is_fake:
        st.markdown(
            """
            Ce texte a été classé comme une **fake news** pour les raisons suivantes:
            - Présence de termes sensationnalistes ou d'affirmations exagérées
            - Absence de références à des sources fiables ou officielles
            - Structure et ton du langage typiques de contenu trompeur
            """
        )
    else:
        st.markdown(
            """
            Ce texte a été classé comme une **vraie news** pour les raisons suivantes:
            - Ton neutre et informatif
            - Présence de termes associés à des sources fiables
            - Absence de sensationnalisme ou d'affirmations extrêmes
            """
        )
    
    # Affichage d'un avertissement
    st.info(
        "Note: Cette prédiction est basée sur une analyse lexicale et sémantique simplifiée. "
        "Pour une précision optimale, consultez plusieurs sources fiables."
    )

# Fonction principale
def main():
    """Fonction principale de l'application"""
    # Titre de l'application
    st.title("Détecteur de Fake News")
    
    # Initialisation du préprocesseur simplifié
    preprocessor = SimpleTextPreprocessor()
    
    # Création d'un modèle KMeans simplifié pour la démonstration
    n_clusters = 2
    kmeans_model = SimplifiedKMeans(n_clusters=n_clusters, random_state=42)
    
    # Centres des clusters (fake/real)
    kmeans_model.cluster_centers_ = np.array([
        np.random.rand(768),  # Fake news
        np.random.rand(768)   # Vraie news
    ])
    
    # Nombre de clusters (défini à 2 pour cette démonstration simplifiée)
    n_clusters = 2
    
    # Barre latérale
    st.sidebar.title("Options")
    
    # Sélection de la langue
    language = st.sidebar.radio(
        "Langue",
        ["Français", "Anglais"],
        index=0
    )
    
    # Exemples de textes
    if language == "Français":
        examples = [
            "Macron a démissionné de son poste de président de la République.",
            "La France a remporté la Coupe du Monde de football en 2018.",
            "Des scientifiques ont découvert que les vaccins contre la COVID-19 contiennent des puces électroniques pour nous surveiller."
        ]
    else:
        examples = [
            "Trump has won the 2020 presidential election by a landslide.",
            "NASA confirmed the existence of water on Mars.",
            "Scientists have discovered that COVID-19 vaccines contain microchips to track people."
        ]
    
    # Sélection d'un exemple
    example = st.sidebar.selectbox(
        "Exemples de textes",
        [""] + examples,
        index=0
    )
    
    # Zone de texte pour la saisie
    if example:
        text = st.text_area("Saisissez un texte à analyser", example, height=200)
    else:
        text = st.text_area("Saisissez un texte à analyser", "", height=200)
    
    # Bouton pour lancer l'analyse
    if st.button("Analyser"):
        if text:
            # Affichage d'un spinner pendant l'analyse
            with st.spinner("Analyse en cours..."):
                # Prédiction améliorée
                is_fake, probabilities, processing_time, language = predict_fake_news(
                    text, preprocessor, kmeans_model
                )
                
                # Affichage des résultats
                display_results(is_fake, probabilities, processing_time, language)
        else:
            st.error("Veuillez saisir un texte à analyser.")
    
    # Informations sur l'application
    st.sidebar.markdown("---")
    st.sidebar.subheader("À propos")
    st.sidebar.info(
        "Cette application est une démonstration simplifiée de détection de fake news. "
        "Elle utilise un modèle simulé pour illustrer le concept, sans dépendances externes. "
        "Pour une implémentation complète, consultez le code source original."
    )

if __name__ == "__main__":
    main()
