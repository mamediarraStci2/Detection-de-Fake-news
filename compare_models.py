"""
Script de comparaison des performances entre les modèles transformers et les algorithmes de clustering
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import json
from tabulate import tabulate

RESULTS_DIR = os.path.join("models", "results")

def extract_metrics_from_file(file_path):
    """Extrait les métriques de performance à partir d'un fichier d'évaluation"""
    metrics = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Extraction des métriques communes
            accuracy_match = re.search(r'Accuracy:\s*([\d\.]+)', content)
            f1_match = re.search(r'F1 Score:\s*([\d\.]+)', content)
            precision_match = re.search(r'Precision:\s*([\d\.]+)', content)
            recall_match = re.search(r'Recall:\s*([\d\.]+)', content)
            
            if accuracy_match:
                metrics['Accuracy'] = float(accuracy_match.group(1))
            if f1_match:
                metrics['F1'] = float(f1_match.group(1))
            if precision_match:
                metrics['Precision'] = float(precision_match.group(1))
            if recall_match:
                metrics['Recall'] = float(recall_match.group(1))
                
    except Exception as e:
        print(f"Erreur lors de l'extraction des métriques depuis {file_path}: {e}")
    
    return metrics

def load_csv_results(file_path):
    """Charge les résultats depuis un fichier CSV"""
    try:
        df = pd.read_csv(file_path)
        return df.to_dict('records')
    except Exception as e:
        print(f"Erreur lors du chargement des résultats depuis {file_path}: {e}")
        return []

def compare_models():
    """Compare les performances des différents modèles"""
    results = {
        "transformers": {},
        "clustering": {}
    }
    
    # Recherche des fichiers d'évaluation
    eval_files = glob.glob(os.path.join(RESULTS_DIR, "evaluation_*.txt"))
    
    for file_path in eval_files:
        file_name = os.path.basename(file_path)
        
        # Déterminer le type de modèle
        if "pytorch" in file_name:
            model_type = "transformers"
        else:
            model_type = "clustering"
        
        # Déterminer la langue
        if "english" in file_name:
            language = "english"
        elif "french" in file_name:
            language = "french"
        else:
            language = "multilingual"
        
        # Extraire les métriques
        metrics = extract_metrics_from_file(file_path)
        
        # Stocker les résultats
        if language not in results[model_type]:
            results[model_type][language] = {}
        
        results[model_type][language] = metrics
    
    # Charger les comparaisons directes depuis les CSV si disponibles
    comparison_file = os.path.join(RESULTS_DIR, "transformer_vs_clustering_pytorch.csv")
    if os.path.exists(comparison_file):
        direct_comparison = load_csv_results(comparison_file)
        results["direct_comparison"] = direct_comparison
    
    return results

def print_results_table(results):
    """Affiche les résultats sous forme de tableau"""
    table_data = []
    headers = ["Modèle", "Langue", "Accuracy", "F1 Score", "Precision", "Recall"]
    
    # Ajouter les résultats des transformers
    for language, metrics in results["transformers"].items():
        row = ["Transformers", language]
        for metric in ["Accuracy", "F1", "Precision", "Recall"]:
            row.append(f"{metrics.get(metric, 'N/A'):.4f}" if metric in metrics else "N/A")
        table_data.append(row)
    
    # Ajouter les résultats du clustering
    for language, metrics in results["clustering"].items():
        row = ["Clustering", language]
        for metric in ["Accuracy", "F1", "Precision", "Recall"]:
            row.append(f"{metrics.get(metric, 'N/A'):.4f}" if metric in metrics else "N/A")
        table_data.append(row)
    
    # Afficher le tableau
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Enregistrer le tableau dans un fichier
    with open(os.path.join(RESULTS_DIR, "comparison_table.txt"), "w", encoding="utf-8") as f:
        f.write(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Créer un DataFrame pour faciliter la visualisation
    comparison_df = pd.DataFrame(table_data, columns=headers)
    return comparison_df

def create_comparison_plots(df):
    """Crée des visualisations pour comparer les performances des modèles"""
    # Nettoyer les données
    for col in ["Accuracy", "F1 Score", "Precision", "Recall"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Créer une figure avec plusieurs sous-graphiques
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comparaison des performances: Transformers vs Clustering', fontsize=16)
    
    metrics = ["Accuracy", "F1 Score", "Precision", "Recall"]
    
    for i, (ax, metric) in enumerate(zip(axes.flatten(), metrics)):
        # Créer un dataframe pivotant pour faciliter le tracé
        pivot_df = df.pivot(index="Langue", columns="Modèle", values=metric)
        pivot_df.plot(kind='bar', ax=ax, rot=0)
        ax.set_title(f'Comparaison par {metric}')
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1)
        
        # Ajouter les valeurs sur les barres
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(RESULTS_DIR, "transformers_vs_clustering_comparison.png"), dpi=300)
    
    # Créer un graphique radar pour une vue d'ensemble
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Filtrer pour afficher uniquement les données anglaises pour simplifier
    english_df = df[df["Langue"] == "english"]
    
    # Nombre de variables
    categories = metrics
    N = len(categories)
    
    # Position angulaire de chaque variable sur le cercle
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Fermer le cercle
    
    # Extraire les valeurs pour chaque modèle
    transformers_values = english_df[english_df["Modèle"] == "Transformers"][metrics].values.flatten().tolist()
    transformers_values += transformers_values[:1]  # Fermer le cercle
    
    clustering_values = english_df[english_df["Modèle"] == "Clustering"][metrics].values.flatten().tolist()
    clustering_values += clustering_values[:1]  # Fermer le cercle
    
    # Tracer les lignes et points pour chaque modèle
    ax.plot(angles, transformers_values, 'o-', linewidth=2, label='Transformers')
    ax.fill(angles, transformers_values, alpha=0.25)
    ax.plot(angles, clustering_values, 'o-', linewidth=2, label='Clustering')
    ax.fill(angles, clustering_values, alpha=0.25)
    
    # Définir l'étiquette pour chaque axe
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # Ajouter une légende
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Comparaison des métriques entre Transformers et Clustering (Anglais)')
    plt.savefig(os.path.join(RESULTS_DIR, "radar_comparison.png"), dpi=300)
    
    return fig

def main():
    print("Comparaison des performances entre les modèles transformers et les algorithmes de clustering")
    print("="*80)
    
    # Récupérer les résultats
    results = compare_models()
    
    # Afficher les résultats
    comparison_df = print_results_table(results)
    
    # Créer les visualisations
    create_comparison_plots(comparison_df)
    
    print(f"\nLes résultats et visualisations ont été enregistrés dans le dossier {RESULTS_DIR}")
    print("\nRésumé des avantages et inconvénients:")
    print("\nTransformers (BERT/CamemBERT):")
    print("✓ Généralement de meilleures performances en termes de précision et de rappel")
    print("✓ Prise en compte du contexte sémantique des mots")
    print("✓ Capacité à traiter des textes de différentes langues avec des modèles spécialisés")
    print("✗ Entraînement plus long et nécessitant plus de ressources")
    print("✗ Besoin de données étiquetées pour l'entraînement supervisé")
    
    print("\nClustering (KMeans, etc.):")
    print("✓ Approche non-supervisée ne nécessitant pas de données étiquetées")
    print("✓ Entraînement plus rapide et moins gourmand en ressources")
    print("✓ Utile pour l'exploration de données et la découverte de patterns")
    print("✗ Performances généralement inférieures aux approches supervisées")
    print("✗ Manque de prise en compte du contexte sémantique")

if __name__ == "__main__":
    main()
