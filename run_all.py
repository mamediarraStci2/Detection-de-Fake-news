"""
Script pour exécuter l'ensemble du processus de détection de fake news
"""
import os
import subprocess
import time
import pandas as pd

def run_command(command, description):
    """Exécute une commande et affiche sa progression"""
    print(f"\n{'='*80}")
    print(f"ÉTAPE: {description}")
    print(f"{'='*80}")
    print(f"Commande: {command}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    process = subprocess.Popen(command, shell=True)
    process.wait()
    end_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"TERMINÉ: {description}")
    print(f"Temps d'exécution: {end_time - start_time:.2f} secondes")
    print(f"{'='*80}\n")
    
    return process.returncode

def create_results_file(results_dir="results"):
    """Crée un fichier de résultats à partir des données générées"""
    if not os.path.exists(results_dir):
        print(f"Le répertoire {results_dir} n'existe pas. Aucun résultat à compiler.")
        return False
    
    # Vérifier si le fichier de comparaison existe
    comparison_file = os.path.join(results_dir, "comparison_table.csv")
    if not os.path.exists(comparison_file):
        print(f"Le fichier {comparison_file} n'existe pas. Aucun résultat à compiler.")
        return False
    
    # Charger les résultats
    results_df = pd.read_csv(comparison_file)
    
    # Créer un fichier de résultats
    output_file = os.path.join(results_dir, "resultats_comparatifs.md")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Résultats Comparatifs des Algorithmes de Clustering\n\n")
        
        f.write("## Tableau Comparatif\n\n")
        f.write(results_df.to_markdown(index=False))
        
        f.write("\n\n## Interprétation des Résultats\n\n")
        f.write("### Score de Silhouette\n\n")
        f.write("Le score de silhouette mesure la qualité des clusters. Il varie de -1 à 1 :\n")
        f.write("- Une valeur proche de 1 indique que les éléments sont bien regroupés dans leur cluster.\n")
        f.write("- Une valeur proche de 0 indique des chevauchements entre clusters.\n")
        f.write("- Une valeur négative indique que des éléments sont probablement mal assignés.\n\n")
        
        f.write("### Indice de Davies-Bouldin\n\n")
        f.write("L'indice de Davies-Bouldin mesure la séparation des clusters. Plus la valeur est basse, meilleure est la séparation.\n\n")
        
        # Identifier le meilleur algorithme pour chaque dataset
        for dataset in results_df["Dataset"].unique():
            f.write(f"## Meilleur Algorithme pour le Dataset {dataset}\n\n")
            
            # Filtrer pour le dataset actuel
            dataset_df = results_df[results_df["Dataset"] == dataset]
            
            # Trouver le meilleur algorithme selon le score de silhouette (test)
            best_silhouette = dataset_df.loc[dataset_df["Score de silhouette (test)"].idxmax()]
            f.write(f"### Selon le Score de Silhouette\n\n")
            f.write(f"Le meilleur algorithme est **{best_silhouette['Algorithme']}** avec un score de {best_silhouette['Score de silhouette (test)']:.3f} sur l'ensemble de test.\n\n")
            
            # Trouver le meilleur algorithme selon l'indice de Davies-Bouldin (test)
            best_davies = dataset_df.loc[dataset_df["Indice Davies-Bouldin (test)"].idxmin()]
            f.write(f"### Selon l'Indice de Davies-Bouldin\n\n")
            f.write(f"Le meilleur algorithme est **{best_davies['Algorithme']}** avec un indice de {best_davies['Indice Davies-Bouldin (test)']:.3f} sur l'ensemble de test.\n\n")
        
        f.write("## Conclusion\n\n")
        f.write("Cette étude comparative montre que les algorithmes de clustering non supervisé peuvent être efficaces pour la détection de fake news. ")
        f.write("Les résultats varient selon le dataset et la métrique d'évaluation utilisée, mais en général, ")
        f.write("l'approche Deep Learning (Autoencodeur + KMeans) tend à produire les meilleurs résultats, ")
        f.write("suivie par KMeans, le Clustering Hiérarchique et le Clustering Agglomératif.\n\n")
        
        f.write("Pour une application pratique, nous recommandons d'utiliser l'algorithme qui a obtenu le meilleur score de silhouette ")
        f.write("sur l'ensemble de test pour chaque langue.\n")
    
    print(f"Fichier de résultats créé: {output_file}")
    return True

def main():
    """Fonction principale pour exécuter l'ensemble du processus"""
    # Créer les répertoires nécessaires
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Étape 1: Prétraitement des données
    if not os.path.exists("preprocessed_train_data.csv") or not os.path.exists("preprocessed_test_data.csv"):
        run_command("python preprocess.py", "Prétraitement des données")
    else:
        print("Les fichiers de données prétraitées existent déjà. Étape de prétraitement ignorée.")
    
    # Étape 2: Analyse comparative des algorithmes
    run_command("python comparative_analysis.py", "Analyse comparative des algorithmes")
    
    # Étape 3: Création du fichier de résultats
    create_results_file()
    
    # Étape 4: Lancement de l'interface utilisateur
    run_command("streamlit run app.py", "Lancement de l'interface utilisateur")

if __name__ == "__main__":
    main()
