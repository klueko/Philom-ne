import os
import json
import re
import pandas as pd
import shutil
import requests
from datetime import datetime, timedelta
from unidecode import unidecode
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# ===================== Configuration =====================
DATA_FILE = "reponse.json"              # Fichier JSON source
FAISS_INDEX_PATH = "faiss_index"        # Répertoire pour stocker l'index FAISS

# ===================== PARTIE 1 : Chargement des données =====================
def load_data_as_dataframe():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.json_normalize(data)
    if "firstdate_begin" in df.columns:
        # Conversion en datetime et suppression du fuseau horaire pour obtenir des tz-naïves
        df['firstdate_begin'] = pd.to_datetime(df['firstdate_begin'], errors="coerce").dt.tz_localize(None)
    return df

def load_preprocessed_data():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# ===================== PARTIE 1bis : Extraction des contraintes d'âge =====================
def extract_age_constraints(query):
    """
    Extrait les contraintes d'âge depuis la requête.
    - Pour "minimum X", renvoie (X, None) signifiant que l'événement doit avoir un âge minimum >= X.
    - Pour "maximum X", renvoie (None, X) signifiant que l'événement doit avoir un âge maximum <= X.
    - Pour "enfant de X", renvoie (X, X) pour une contrainte exacte.
    Retourne (None, None) sinon.
    """
    query_norm = unidecode(query.lower())
    min_match = re.search(r"minimum\s+(\d+)", query_norm)
    max_match = re.search(r"maximum\s+(\d+)", query_norm)
    exact_match = re.search(r"enfant(?:s)?\s+de\s+(\d+)", query_norm)
    if min_match:
        return int(min_match.group(1)), None
    elif max_match:
        return None, int(max_match.group(1))
    elif exact_match:
        age = int(exact_match.group(1))
        return age, age
    return None, None

# ===================== PARTIE 2 : Création et chargement de l'index FAISS =====================
def create_faiss_db(data):
    texts = []
    for event in data:
        date_iso = event.get("firstdate_begin", "")[:10]  # On suppose "YYYY-MM-DD"
        try:
            dt = pd.to_datetime(date_iso, errors="coerce")
            date_str = "" if pd.isna(dt) else dt.strftime("%Y-%m-%d")
        except Exception:
            date_str = ""
        texte = (
            f"Date: {date_str}\n"
            f"Mois: {date_str[5:7] if date_str else ''}\n"
            f"Année: {date_str[:4] if date_str else ''}\n"
            f"Title: {event.get('title_fr', '')}\n"
            f"Description: {event.get('description_fr', '')}\n"
            f"Conditions: {event.get('conditions_fr', '')}"
        )
        texts.append(texte)
    documents = [Document(page_content=t) for t in texts]
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embedder)
    vector_store.save_local(FAISS_INDEX_PATH)
    print(f"✅ Index FAISS créé et sauvegardé dans '{FAISS_INDEX_PATH}'.")
    return vector_store

def load_faiss():
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(FAISS_INDEX_PATH, embedder, allow_dangerous_deserialization=True)
    return vector_store.as_retriever()

# ===================== PARTIE 3 : Extraction de la date =====================
def extract_date_from_text(document_text):
    pattern = r"Date:\s*(\d{4}-\d{2}-\d{2})"
    match = re.search(pattern, document_text)
    if match:
        return pd.to_datetime(match.group(1), errors="coerce")
    return None

# ===================== PARTIE 4 : Extraction des informations de date depuis la requête =====================
def extract_date_from_query(query):
    query_norm = unidecode(query.lower())
    if "date:" in query_norm:
        match_full = re.search(r"date:\s*(\d{4}-\d{2}-\d{2})", query_norm)
        if match_full:
            full_date = match_full.group(1)
            dt = pd.to_datetime(full_date, errors="coerce")
            if dt:
                return dt.day, dt.month, dt.year
        match_partial = re.search(r"date:\s*(\d{4}-\d{2})", query_norm)
        if match_partial:
            date_part = match_partial.group(1)
            annee_str, mois_str = date_part.split("-")
            return None, int(mois_str), int(annee_str)
    match_day = re.search(r"(\d{1,2})\s+(janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre)\s*(\d{4})?",
                          query_norm)
    if match_day:
        day = int(match_day.group(1))
        mois_str = match_day.group(2)
        months = {"janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4,
                  "mai": 5, "juin": 6, "juillet": 7, "août": 8, "aout": 8,
                  "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12, "decembre": 12}
        mois = months.get(mois_str)
        annee = int(match_day.group(3)) if match_day.group(3) else 2025
        return day, mois, annee
    pattern = r"(janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre)"
    month_match = re.search(pattern, query_norm)
    if month_match:
        mois_str = month_match.group(0)
        months = {"janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4,
                  "mai": 5, "juin": 6, "juillet": 7, "août": 8, "aout": 8,
                  "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12, "decembre": 12}
        mois = months.get(mois_str)
        year_match = re.search(r"\b(\d{4})\b", query_norm)
        annee = int(year_match.group(1)) if year_match else 2025
        return None, mois, annee
    year_match = re.search(r"\b(\d{4})\b", query_norm)
    if year_match:
        annee = int(year_match.group(1))
        return None, -1, annee
    return None, None, None

# ===================== PARTIE 4bis : Extraction des dates relatives =====================
def extract_relative_date(query):
    query_norm = unidecode(query.lower())
    today = datetime.today()
    
    # Gestion explicite pour "aujourd'hui"
    if "aujourd'hui" in query_norm or "auj" in query_norm:
        start_today = datetime(today.year, today.month, today.day)
        end_today = start_today + timedelta(days=1) - timedelta(seconds=1)
        print("DEBUG (aujourd'hui):", start_today.strftime("%Y-%m-%d"), "to", end_today.strftime("%Y-%m-%d"))
        return start_today, end_today
    
    # Gestion explicite pour "demain"
    if "demain" in query_norm:
        tomorrow = today + timedelta(days=1)
        start_tomorrow = datetime(tomorrow.year, tomorrow.month, tomorrow.day)
        end_tomorrow = start_tomorrow + timedelta(days=1) - timedelta(seconds=1)
        print("DEBUG (demain):", start_tomorrow.strftime("%Y-%m-%d"), "to", end_tomorrow.strftime("%Y-%m-%d"))
        return start_tomorrow, end_tomorrow

    if "cette semaine" in query_norm:
        days_until_sunday = 6 - today.weekday()
        week_end = today + timedelta(days=days_until_sunday)
        print("DEBUG (cette semaine):", today.strftime("%Y-%m-%d"), "to", week_end.strftime("%Y-%m-%d"))
        return today, week_end
    if "semaine prochaine" in query_norm:
        days_until_monday = 7 - today.weekday() if today.weekday() != 0 else 7
        next_monday = today + timedelta(days=days_until_monday)
        next_sunday = next_monday + timedelta(days=6)
        print("DEBUG (semaine prochaine):", next_monday.strftime("%Y-%m-%d"), "to", next_sunday.strftime("%Y-%m-%d"))
        return next_monday, next_sunday
    if "mois prochain" in query_norm:
        if today.month == 12:
            next_month = 1
            year = today.year + 1
        else:
            next_month = today.month + 1
            year = today.year
        start_date = datetime(year, next_month, 1)
        if next_month == 12:
            next_month_start = datetime(year + 1, 1, 1)
        else:
            next_month_start = datetime(year, next_month + 1, 1)
        end_date = next_month_start - timedelta(days=1)
        print("DEBUG (mois prochain):", start_date.strftime("%Y-%m-%d"), "to", end_date.strftime("%Y-%m-%d"))
        return start_date, end_date
    # Gestion du weekend (weekend prochain, ce weekend, le weekend, etc.)
    if "weekend prochain" in query_norm or "week end prochain" in query_norm or "week-end prochain" in query_norm or \
       "ce weekend" in query_norm or "le weekend" in query_norm or "ce week end" in query_norm or "ce week-end" in query_norm:
        if today.weekday() < 5:
            days_to_saturday = 5 - today.weekday()
            saturday = today + timedelta(days=days_to_saturday)
            sunday = saturday + timedelta(days=1)
        else:
            days_to_next_saturday = (7 - today.weekday()) + 5
            saturday = today + timedelta(days=days_to_next_saturday)
            sunday = saturday + timedelta(days=1)
        print("DEBUG (weekend):", saturday.strftime("%Y-%m-%d"), "to", sunday.strftime("%Y-%m-%d"))
        return saturday, sunday
    match = re.search(r"dans (\d+) jours", query_norm)
    if match:
        nb_jours = int(match.group(1))
        target_date = today + timedelta(days=nb_jours)
        print("DEBUG (dans X jours):", target_date.strftime("%Y-%m-%d"))
        return target_date, target_date
    return None, None

# ===================== PARTIE 5ter : Filtrage par catégorie =====================
def filter_by_category(results, query):
    query_norm = unidecode(query.lower())
    category = None
    keywords = []
    if "concert" in query_norm:
        category = "concert"
        keywords = ["concert", "live", "band", "musique", "chœur", "orchestrale"]
    elif "atelier" in query_norm:
        category = "atelier"
        keywords = ["atelier", "stage", "cours"]
    elif "exposition" in query_norm or "exposition d'art" in query_norm:
        category = "exposition d'art"
        keywords = ["exposition", "galerie", "art", "peinture", "sculpture"]
    elif "conférence" in query_norm:
        category = "conférences"
        keywords = ["conférence", "table ronde", "débats"]
    elif "rendez-vous" in query_norm or "soirée" in query_norm or "meeting" in query_norm:
        category = "rendez-vous sociaux"
        keywords = ["rendez-vous", "soirée", "meeting", "forum", "réunion"]

    if not category:
        return results

    filtered = []
    for text in results:
        if any(keyword in text.lower() for keyword in keywords):
            filtered.append(text)
    return filtered

# ===================== PARTIE 5bis : Recherche hybride combinant préfiltrage pandas, filtrage par catégorie ET FAISS =====================
def search_events(query, retriever, df):
    day, target_month, target_year = extract_date_from_query(query)
    # Utiliser la date relative si aucune date explicite n'est trouvée
    if target_month is None or target_year is None:
        rel_start, rel_end = extract_relative_date(query)
        if rel_start and rel_end:
            target_month = rel_start.month
            target_year = rel_start.year
            # Conserver day None pour utiliser l'intervalle complet
            day = None

    if target_month is not None and target_year is not None:
        # Si une date relative a été détectée, utiliser l'intervalle complet
        if extract_relative_date(query)[0] is not None:
            rel_start, rel_end = extract_relative_date(query)
            df_filtered = df[(df['firstdate_begin'] >= rel_start) & (df['firstdate_begin'] <= rel_end)]
            crit_str = f"entre {rel_start.strftime('%Y-%m-%d')} et {rel_end.strftime('%Y-%m-%d')}"
        elif target_month == -1:
            df_filtered = df[df['firstdate_begin'].dt.year == target_year]
            crit_str = f"année {target_year}"
        elif day is not None:
            df_filtered = df[(df['firstdate_begin'].dt.year == target_year) &
                             (df['firstdate_begin'].dt.month == target_month) &
                             (df['firstdate_begin'].dt.day == day)]
            crit_str = f"{day}/{target_month}/{target_year}"
        else:
            df_filtered = df[(df['firstdate_begin'].dt.year == target_year) &
                             (df['firstdate_begin'].dt.month == target_month)]
            crit_str = f"{target_month}/{target_year}"
            
        if not df_filtered.empty:
            # Filtrage par contrainte d'âge
            age_min_constraint, age_max_constraint = extract_age_constraints(query)
            if age_min_constraint is not None or age_max_constraint is not None:
                df_filtered = df_filtered[
                    df_filtered['age_min'].notna() & df_filtered['age_max'].notna()
                ]
                if age_min_constraint is not None and age_max_constraint is not None:
                    df_filtered = df_filtered[
                        (df_filtered['age_min'] <= age_min_constraint) &
                        (df_filtered['age_max'] >= age_max_constraint)
                    ]
                elif age_min_constraint is not None:
                    df_filtered = df_filtered[
                        (df_filtered['age_min'] >= age_min_constraint)
                    ]
                elif age_max_constraint is not None:
                    df_filtered = df_filtered[
                        (df_filtered['age_max'] <= age_max_constraint)
                    ]
                crit_str += f", pour un enfant avec contrainte d'âge"
            print(f"\n✅ Préfiltrage pandas : {len(df_filtered)} événements trouvés pour {crit_str}")
            results = []
            for idx, row in df_filtered.iterrows():
                date_str = row['firstdate_begin'].strftime("%Y-%m-%d") if pd.notnull(row['firstdate_begin']) else ""
                texte = (
                    f"Date: {date_str}\n"
                    f"Title: {row.get('title_fr','')}\n"
                    f"Description: {row.get('description_fr','')}\n"
                    f"Conditions: {row.get('conditions_fr','')}"
                )
                results.append(texte)
            results = filter_by_category(results, query)
            if results:
                return results
            else:
                print("\n❌ Aucun événement trouvé après filtrage par catégorie via pandas.")
    
    try:
        docs = retriever.search(query, k=100)
    except Exception:
        docs = retriever.invoke(query)
    results = [doc.page_content for doc in docs]
    if target_month is not None and target_year is not None and target_month != -1:
        print(f"\n🎯 Filtrage explicite FAISS des événements pour {target_month}/{target_year}")
        filtered_results = []
        for text in results:
            doc_date = extract_date_from_text(text)
            if doc_date:
                if day is not None:
                    if (doc_date.year == target_year and 
                        doc_date.month == target_month and 
                        doc_date.day == day):
                        filtered_results.append(text)
                else:
                    if (doc_date.year == target_year and 
                        doc_date.month == target_month):
                        filtered_results.append(text)
        if not filtered_results:
            return [f"❌ Aucun événement trouvé via FAISS pour {target_month}/{target_year}."]
        results = filtered_results
    elif target_year is not None and target_month == -1:
        print(f"\n🎯 Filtrage explicite FAISS des événements pour l'année {target_year}")
        filtered_results = []
        for text in results:
            doc_date = extract_date_from_text(text)
            if doc_date and doc_date.year == target_year:
                filtered_results.append(text)
        if not filtered_results:
            return [f"❌ Aucun événement trouvé via FAISS pour l'année {target_year}."]
        results = filtered_results

    results = filter_by_category(results, query)
    return results

# ===================== PARTIE 5bis : Boucle interactive du chatbot =====================
def chatbot_loop(df, retriever):
    print("💬 Tapez 'quit' pour quitter le chatbot.")
    while True:
        query = input("📝 Posez votre question : ")
        if query.lower() in ["quit", "exit"]:
            print("👋 Au revoir.")
            break
        results = search_events(query, retriever, df)
        print("\n🔍 Résultats obtenus :")
        for r in results:
            print("-" * 40)
            print(r)
            print("-" * 40)
        print("\n")
        prompt = "Voici les événements correspondant à votre recherche :\n\n"
        for event in results:
            prompt += f"- {event}\n"
        prompt += "\nPouvez-vous résumer ces événements et donner des informations utiles ?"
        ollama_response = query_ollama(prompt)
        print("\n💬 Réponse d'Ollama :\n", ollama_response)
    print("\n")
# ===================== PARTIE 5ter : Fonction query_ollama =====================
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
                    continue

        if not final_response:
            return "Aucune réponse obtenue."

        # Formater la réponse en HTML
        formatted_response = final_response.replace("\n", "<br>").replace("**", "<b>")
        return formatted_response

    except requests.exceptions.RequestException as e:
        return f"❌ Erreur de connexion à Ollama : {e}"
# ===================== PARTIE 6 : Exécution principale =====================
if __name__ == "__main__":
    print("🚀 Début du processus...")
    df = load_data_as_dataframe()
    print(f"✅ {len(df)} enregistrements chargés depuis '{DATA_FILE}' (DataFrame).")
    data = load_preprocessed_data()
    if os.path.exists(FAISS_INDEX_PATH):
        print("ℹ️ Index FAISS existant, chargement...")
        retriever = load_faiss()
    else:
        print("ℹ️ Aucun index FAISS trouvé, création...")
        create_faiss_db(data)
        retriever = load_faiss()
    chatbot_loop(df, retriever)