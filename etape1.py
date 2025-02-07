import os
import json
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# 📌 1. Récupération des données depuis l'API Bordeaux Métropole
URL = "https://datahub.bordeaux-metropole.fr/api/explore/v2.1/catalog/datasets/met_agenda/records"
DATA_FILE = "reponse.json"
FAISS_INDEX_PATH = "faiss_index"

import time  # Ajout pour gérer les retries en cas d'échec

def fetch_data():
    """
    Récupère les données de Bordeaux Métropole et les sauvegarde en JSON.
    Gère les erreurs réseau et les retries en cas d'échec.
    """
    all_data = []
    offset = 0
    limit = 100  # Nombre de résultats par requête
    max_records = 10000  # Nombre total de résultats à récupérer
    max_retries = 5  # Nombre maximum de tentatives en cas d'échec

    while len(all_data) < max_records:
        params = {'limit': limit, 'offset': offset}
        retries = 0

        while retries < max_retries:
            try:
                response = requests.get(URL, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json().get('results', [])
                    if not data:
                        print("✅ Fin des données (aucune nouvelle entrée trouvée).")
                        break  # Arrêter si aucune donnée retournée
                    all_data.extend(data)
                    offset += limit
                    break  # Sortir de la boucle de retry après succès

                else:
                    print(f"⚠️ Erreur HTTP {response.status_code}, tentative {retries + 1}/{max_retries}")
                    retries += 1
                    time.sleep(2)  # Attendre avant de réessayer

            except requests.exceptions.RequestException as e:
                print(f"⚠️ Erreur réseau : {e}, tentative {retries + 1}/{max_retries}")
                retries += 1
                time.sleep(2)  # Pause avant le retry

        if retries == max_retries:
            print(f"❌ Échec après {max_retries} tentatives. Arrêt de la récupération.")
            break  # Arrête complètement si trop d'échecs

    # Sauvegarde des données récupérées
    with open(DATA_FILE, "w", encoding="utf-8") as file:
        json.dump(all_data, file, indent=4, ensure_ascii=False)

    print(f"✅ {len(all_data)} événements enregistrés dans {DATA_FILE}")


def clean_and_filter_data_bis():
    """
    Charge et filtre les données, en supprimant les colonnes inutiles et en filtrant les événements après le 1er janvier 2025.
    """
    with open(DATA_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    df = pd.json_normalize(data)

    # Sélection des colonnes utiles
    columns_to_keep = ['title_fr', 'description_fr', 'conditions_fr', 'firstdate_begin']
    df = df[columns_to_keep].dropna()

    # ✅ Vérification des données brutes AVANT filtrage
    print("\n📌 Vérification des événements AVANT filtrage :")
    print(df[['title_fr', 'firstdate_begin']].head(20))  # Afficher les 20 premiers événements

    # Convertir la date et filtrer après le 1er janvier 2025
    df['firstdate_begin'] = pd.to_datetime(df['firstdate_begin'], errors='coerce')

def clean_and_filter_data():
    """
    Charge et filtre les données, en supprimant les colonnes inutiles et en filtrant les événements après le 1er janvier 2025.
    """
    with open(DATA_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    df = pd.json_normalize(data)

    # Sélection des colonnes utiles
    columns_to_keep = ['title_fr', 'description_fr', 'conditions_fr', 'firstdate_begin']
    df = df[columns_to_keep].dropna()

    # ✅ Conversion des dates
    df['firstdate_begin'] = pd.to_datetime(df['firstdate_begin'], errors='coerce')

    print(df.head(10))

    # ✅ Vérification des dates avant filtrage
    invalid_dates = df[df['firstdate_begin'].isna()]
    if not invalid_dates.empty:
        print("\n⚠️ Événements avec une date NON VALIDE :")
        print(invalid_dates[['title_fr', 'firstdate_begin']])
        print(f"❌ {len(invalid_dates)} événements ignorés à cause d'une date invalide.")

    # ✅ Vérification des événements supprimés après filtrage
    excluded_events = df[df['firstdate_begin'] <= "2025-01-01"]
    if not excluded_events.empty:
        print("\n⚠️ Événements EXCLUS après filtrage (dates ≤ 2025-01-01) :")
        print(excluded_events[['title_fr', 'firstdate_begin']])
        print(f"❌ {len(excluded_events)} événements supprimés après filtrage.")

    # ✅ Modification du filtrage pour éviter de supprimer des événements incorrectement
    df = df[(df['firstdate_begin'].notna()) & (df['firstdate_begin'] >= "2025-01-01")]

    if df.empty:
        print("❌ Aucun événement trouvé après le 1er janvier 2025.")
        exit()

    # ✅ Vérification des événements restants par mois après filtrage
    df['mois'] = df['firstdate_begin'].dt.month
    df['année'] = df['firstdate_begin'].dt.year

    print("\n📊 Nombre d'événements par mois après filtrage :")
    print(df.groupby(['année', 'mois']).size())

    # Supprime les colonnes temporaires après affichage
    df.drop(columns=['mois', 'année'], inplace=True)

    print(f"✅ {len(df)} événements conservés après filtrage.")
    return df

def create_faiss_db(df):
    """
    Génère les embeddings et stocke les documents dans FAISS avec la date comme métadonnée.
    """
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Vérifier que df contient bien des événements
    if df.empty:
        print("❌ Aucun événement à indexer dans FAISS.")
        return

    texts = df.apply(lambda row: f"{row['title_fr']} - {row['description_fr']} ({row['conditions_fr']}, {row['firstdate_begin']})", axis=1).tolist()
    dates = df['firstdate_begin'].astype(str).tolist()  # Convertir les dates en texte

    print(f"✅ {len(texts)} événements prêts pour l'indexation FAISS.")

    # Générer les embeddings
    embeddings = model.encode(texts, convert_to_tensor=False)
    print(f"✅ {len(embeddings)} embeddings générés avec succès.")

    # Création des documents LangChain pour FAISS avec métadonnées
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    documents = [Document(page_content=text, metadata={"date": date}) for text, date in zip(texts, dates)]

    # Vérification avant indexation
    print(f"✅ {len(documents)} documents FAISS créés avec métadonnées.")

    # Indexation dans FAISS
    vector_store = FAISS.from_documents(documents, embedder)
    vector_store.save_local(FAISS_INDEX_PATH)

    print(f"✅ Base FAISS enregistrée avec {len(documents)} événements.")

if __name__ == "__main__":
    print("🚀 Début du processus de récupération et d'indexation des événements...")
    
    fetch_data()  # Étape 1 : Récupération des données
    df = clean_and_filter_data()  # Étape 2 : Nettoyage et filtrage
    create_faiss_db(df)  # Étape 3 : Création et sauvegarde de FAISS

    # 🔍 Vérification des événements indexés dans FAISS
    print("\n📌 Vérification des événements indexés dans FAISS :")
    retriever = FAISS.load_local(
        FAISS_INDEX_PATH,
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        allow_dangerous_deserialization=True
    ).as_retriever()  # ✅ Important pour utiliser FAISS en mode recherche

    query_test = "mars 2025"
    docs = retriever.get_relevant_documents(query_test)  # ✅ Correct

    if docs:
        print(f"✅ {len(docs)} événements trouvés pour '{query_test}' dans FAISS.")
        for doc in docs[:10]:  # 🔍 Affiche les 10 premiers résultats
            print(doc.page_content)
    else:
        print(f"❌ Aucun événement trouvé pour la requête : {query_test}")

    print("🎉 Processus terminé avec succès !")
