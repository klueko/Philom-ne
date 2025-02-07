# Chatbot d'Histoire

Ce projet crée un chatbot interactif qui recommende des sorties culturels à Bordeaux. Le chatbot est construit avec **Flask** pour la gestion du backend, une base de données SQLite pour stocker certaines données, et une interface utilisateur en **JavaScript**.

## Technologies utilisées

- **Backend (Python)**
  - [Flask](https://flask.palletsprojects.com/)
  - [SQLite](https://www.sqlite.org/)
  - **LangChain** pour intégrer des modèles d'IA et gérer les embeddings
  - **FAISS** pour l'indexation et la recherche rapide d'événements basés sur des embeddings

- **Frontend (JavaScript)**
  - [Node.js](https://nodejs.org/)
  - **HTML/CSS** pour l'interface utilisateur

- **Traitement du langage naturel (NLP)**
  - **HuggingFace et all-MiniLM-L6-v2** pour la génération des embeddings des événements
  - **Ollama** pour générer des résumés et des réponses aux questions de l'utilisateur.

## Installation

### Prérequis

- Python 3.x
- Node.js et npm
- Une base de données SQLite
- Les modèles NLP comme **MiniLM** pour l'encodage de texte et l'indexation dans **FAISS**

### Backend (Flask)

1. Clonez le repository :
    ```bash
    git clone https://github.com/klueko/histerIA.git
    cd Philomene
    ```

2. Créez un environnement virtuel Python :
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Sur Mac/Linux
    venv\Scripts\activate     # Sur Windows
    ```

3. Installez les dépendances Python :
    ```bash
    pip install -r requirements.txt
    ```

4. Configurez la base de données (SQLite) :
    - Le fichier `reponse.json` contenant les événements est nécessaire pour initialiser la base de données. Assurez-vous que ce fichier est dans le répertoire approprié.

5. Lancez le backend Flask :
    ```bash
    python app.py
    ```

    Le backend sera accessible à `http://localhost:5000`.

### Fonctionnalité

- **Recherche rapide avec FAISS** : Utilisation de l'indexation vectorielle pour rechercher des événements similaires en fonction des embeddings générés.
- **Résumé automatique** : Lorsque l'utilisateur interagit, Ollama génère des résumés ou des informations supplémentaires sur les événements pertinents.


