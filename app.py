from flask import Flask, render_template, request, jsonify
import json
import pandas as pd
from etape2_2 import load_faiss, search_events, query_ollama

app = Flask(__name__)

# Charger le FAISS retriever et les données au démarrage de l'application
retriever = load_faiss()
with open("reponse.json", "r", encoding="utf-8") as file:
    data = json.load(file)
df = pd.json_normalize(data)
df['firstdate_begin'] = pd.to_datetime(df['firstdate_begin'], errors='coerce')

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"answer": "Je n'ai pas compris votre question."})

    # Détecter les messages de salutation
    greetings = ["bonjour", "salut", "coucou"]
    if user_message.lower() in greetings:
        return jsonify({"answer": "Bonjour ! Comment puis-je vous aider aujourd'hui ?"})

    # Utiliser les fonctions de etape2.py pour obtenir les résultats
    search_results = search_events(user_message, retriever, df)
    if search_results and "❌" not in search_results[0]:
        prompt = "Voici les événements correspondant à votre recherche :\n\n"
        for event in search_results:
            prompt += f"- {event}\n"
        prompt += "\nPouvez-vous résumer ces événements et donner des informations utiles ?"

        response = query_ollama(prompt)
        return jsonify({"answer": response})
    else:
        return jsonify({"answer": search_results[0]})

if __name__ == '__main__':
    app.run(debug=True)