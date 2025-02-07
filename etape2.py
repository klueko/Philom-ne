import json
import requests
import re
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

FAISS_INDEX_PATH = "faiss_index"

def load_faiss():
    """
    Charge la base de données vectorielle FAISS.
    """
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(FAISS_INDEX_PATH, embedder, allow_dangerous_deserialization=True)
    return vector_store.as_retriever()

def extract_date_from_query(query):
    """
    Extrait une date (mois et année) à partir de la requête utilisateur.
    """
    mois_dict = {
        "janvier": 1, "février": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
        "juillet": 7, "août": 8, "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12
    }

    pattern = r"\b(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+(\d{4})\b"
    match = re.search(pattern, query.lower())

    if match:
        mois = mois_dict.get(match.group(1))  # Récupère le numéro du mois
        annee = int(match.group(2))  # Récupère l'année
        return mois, annee, match.group(0)  # Retourne aussi "mai 2025" par ex.

    return None, None, None  # Si aucune date trouvée

def extract_date_from_text(event):
    """
    Extrait une date correcte depuis un événement FAISS.
    """
    pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
    match = re.search(pattern, event)
    
    if match:
        return pd.to_datetime(match.group(0), errors='coerce')
    
    return None  # Si aucune date trouvée

def search_events(query, retriever, df):
    """
    Recherche les événements avec FAISS et applique un filtrage strict par date après récupération.
    """
    mois, annee, date_text = extract_date_from_query(query)

    # 🔥 Étape 1 : Améliorer la requête pour FAISS
    if mois and annee:
        faiss_query = f"Événements en {date_text} à Bordeaux"
    else:
        faiss_query = query  # Si pas de date détectée, requête classique

    print(f"\n🔍 Envoi de la requête à FAISS : {faiss_query}")

    # 🔥 Étape 2 : Premier appel FAISS
    docs = retriever.invoke(faiss_query)
    results = [doc.page_content for doc in docs]

    # 🔍 Debugging : Voir les 10 premiers résultats avant filtrage
    print("\n🔍 Résultats FAISS avant filtrage :")
    for event in results[:10]:
        print(event)

    # 🔥 Étape 3 : Deuxième essai si FAISS retourne peu de résultats
    if len(results) < 3:
        faiss_query = f"Agenda événements {date_text} Bordeaux"
        print(f"\n🔍 Deuxième tentative FAISS : {faiss_query}")
        docs = retriever.invoke(faiss_query)
        results.extend([doc.page_content for doc in docs])

    # 🛑 Si pas de date demandée, retourner les résultats bruts FAISS
    if not (mois and annee):
        return results  

    # 🎯 Étape 4 : Filtrage strict par date avec Pandas
    filtered_df = df[
        (df['firstdate_begin'].dt.month == mois) &
        (df['firstdate_begin'].dt.year == annee)
    ]

    if filtered_df.empty:
        return [f"❌ Aucun événement trouvé pour '{date_text}'"]

    # 🏆 Étape 5 : Récupérer les événements filtrés et les retourner
    final_results = filtered_df.apply(
        lambda row: f"{row['title_fr']} - {row['description_fr']} ({row['conditions_fr']}, {row['firstdate_begin']})",
        axis=1
    ).tolist()

    return final_results

def query_ollama(prompt, model="llama3.2"):
    """
    Envoie un prompt à Ollama et récupère une réponse complète en gérant le streaming.
    """
    payload = {"model": model, "prompt": prompt}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload, headers=headers, stream=True)

        if response.status_code != 200:
            return f"❌ Erreur : {response.status_code} - {response.text}"

        # Lire la réponse en streaming et reconstituer la réponse complète
        final_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    json_data = json.loads(line.decode("utf-8"))
                    final_response += json_data.get("response", "")
                except json.JSONDecodeError:
                    continue  # Ignore les erreurs de parsing

        return final_response if final_response else "Aucune réponse obtenue."

    except requests.exceptions.RequestException as e:
        return f"❌ Erreur de connexion à Ollama : {e}"

if __name__ == "__main__":
    retriever = load_faiss()

    # Charger les données Pandas pour filtrage par date
    with open("reponse.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    df = pd.json_normalize(data)
    df['firstdate_begin'] = pd.to_datetime(df['firstdate_begin'], errors='coerce')

    # Demander une requête utilisateur
    user_query = input("📝 Posez votre question : ")
    search_results = search_events(user_query, retriever, df)

    # Affichage des résultats
    if search_results and "❌" not in search_results[0]:  # Vérifier s'il y a des résultats valides
        prompt = "Voici les événements correspondant à votre recherche :\n\n"
        for event in search_results:
            prompt += f"- {event}\n"
        prompt += "\nPouvez-vous résumer ces événements et donner des informations utiles ?"

        response = query_ollama(prompt)
        print("\n💬 Réponse d'Ollama :\n", response)
    else:
        print("\n💬 Réponse d'Ollama :", search_results[0])