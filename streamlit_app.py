"""
Point d'entrée principal pour le déploiement sur Streamlit Cloud
"""
import streamlit as st
import os
import sys

# Désactiver les avertissements liés à TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Assurez-vous que l'application utilise la version PyTorch des transformers
os.environ['USE_PYTORCH'] = 'true'

# Import notre application principale
from app import main

# Exécution de l'application
if __name__ == "__main__":
    st.set_page_config(
        page_title="Détecteur de Fake News",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("Détecteur de Fake News")
    st.sidebar.info(
        "Ce détecteur de fake news utilise des modèles transformers (PyTorch) et des algorithmes de clustering. "
        "Sélectionnez le type de modèle dans les options ci-dessous."
    )
    main()
