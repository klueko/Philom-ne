import os
import json
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import time  # Pour gérer les retries

# 📌 Configuration
URL = "https://datahub.bordeaux-metropole.fr/api/explore/v2.1/catalog/datasets/met_agenda/records"
DATA_FILE = "reponse.json"
RESPONSE_CLEAN_FILE = "reponses_nettoye.json"
FAISS_INDEX_PATH = "faiss_index"

def fetch_data():
    """
    Récupère les données de Bordeaux Métropole et les sauvegarde dans DATA_FILE (JSON).
    Gère les erreurs réseau et les retries en cas d'échec.
    
    Dans cette version, nous récupérons jusqu'à 25000 enregistrements (ce qui couvre plus de 24219 résultats).
    """
    all_data = []
    offset = 0
    limit = 100         # Nombre de résultats par requête (limite imposée par l'API)
    max_records = 10000 # Nombre total de résultats à récupérer
    max_retries = 5     # Nombre maximum de tentatives en cas d'échec

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
                        break
                    all_data.extend(data)
                    offset += limit
                    break  # Succès : sortir de la boucle de retry
                else:
                    print(f"⚠️ Erreur HTTP {response.status_code}, tentative {retries + 1}/{max_retries}")
                    retries += 1
                    time.sleep(2)
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Erreur réseau : {e}, tentative {retries + 1}/{max_retries}")
                retries += 1
                time.sleep(2)
        if retries == max_retries:
            print(f"❌ Échec après {max_retries} tentatives. Arrêt de la récupération.")
            break

    with open(DATA_FILE, "w", encoding="utf-8") as file:
        json.dump(all_data, file, indent=4, ensure_ascii=False)
    print(f"✅ {len(all_data)} événements enregistrés dans {DATA_FILE}")

def clean_and_filter_data():
    """
    Charge et filtre les données depuis DATA_FILE, en supprimant les colonnes inutiles et 
    en ne conservant que les événements postérieurs au 1er janvier 2025.
    
    Le DataFrame est renommé pour retirer le préfixe 'fields.' (le cas échéant) et le résultat est sauvegardé 
    dans RESPONSE_CLEAN_FILE.
    """
    with open(DATA_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    df = pd.json_normalize(data)
    # Retirer le préfixe "fields." si présent
    df.columns = df.columns.str.replace(r'^fields\.', '', regex=True)
    
    # Liste des colonnes à conserver
    desired_columns = [
        "uid", "title_fr", "description_fr", "conditions_fr", "location_city",
        "keywords_fr", "daterange_fr", "firstdate_begin", "accessibility_label_fr",
        "location_uid", "location_name", "location_address", "location_district",
        "location_postalcode", "location_department", "location_region", "location_countrycode",
        "location_image", "location_imagecredits", "location_phone", "location_website",
        "location_links", "location_description_fr", "location_access_fr",
        "attendancemode", "onlineaccesslink", "status", "age_min", "age_max", "country_fr", "links"
    ]
    # Garder uniquement les colonnes existantes parmi celles désirées
    available_columns = [col for col in desired_columns if col in df.columns]
    df = df[available_columns].dropna(subset=["firstdate_begin"])

    # Conversion des dates en datetime
    df['firstdate_begin'] = pd.to_datetime(df['firstdate_begin'], errors='coerce')

    # Filtrer les événements postérieurs au 1er janvier 2025
    df = df[(df['firstdate_begin'].notna()) & (df['firstdate_begin'] >= "2025-01-01")]
    if df.empty:
        print("❌ Aucun événement trouvé après le 1er janvier 2025.")
        exit()

    # Convertir la date en chaîne pour conserver le format ISO
    df['firstdate_begin'] = df['firstdate_begin'].dt.strftime("%Y-%m-%dT%H:%M:%S%z")

    df.to_json(RESPONSE_CLEAN_FILE, orient="records", indent=4, force_ascii=False)
    print(f"✅ {len(df)} événements nettoyés enregistrés dans {RESPONSE_CLEAN_FILE}")
    return df

def create_faiss_db(df):
    """
    Génère les embeddings et stocke les documents dans FAISS.
    """
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    if df.empty:
        print("❌ Aucun événement à indexer dans FAISS.")
        return

    texts = df.apply(lambda row: f"{row['title_fr']} - {row['description_fr']} ({row['conditions_fr']}, {row['firstdate_begin']})", axis=1).tolist()
    dates = df['firstdate_begin'].astype(str).tolist()

    print(f"✅ {len(texts)} événements prêts pour l'indexation FAISS.")
    embeddings = model.encode(texts, convert_to_tensor=False)
    print(f"✅ {len(embeddings)} embeddings générés avec succès.")

    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    documents = [Document(page_content=text, metadata={"date": date}) for text, date in zip(texts, dates)]
    print(f"✅ {len(documents)} documents FAISS créés avec métadonnées.")

    vector_store = FAISS.from_documents(documents, embedder)
    vector_store.save_local(FAISS_INDEX_PATH)
    print(f"✅ Base FAISS enregistrée avec {len(documents)} événements.")

if __name__ == "__main__":
    print("🚀 Début du processus de récupération et d'indexation des événements...")
    fetch_data()  # Récupération des données
    df = clean_and_filter_data()  # Nettoyage, filtrage et création de RESPONSE_CLEAN_FILE
    create_faiss_db(df)  # Création et sauvegarde de l'index FAISS

    # Vérification des événements indexés dans FAISS
    print("\n📌 Vérification des événements indexés dans FAISS :")
    retriever = FAISS.load_local(
        FAISS_INDEX_PATH,
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        allow_dangerous_deserialization=True
    ).as_retriever()

    print("🎉 Processus terminé avec succès !")
