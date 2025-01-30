# Chatbot d'Histoire

Ce projet crée un chatbot interactif qui raconte des histoires et engage les utilisateurs dans des conversations sur divers thèmes historiques. Le chatbot est construit avec **RASA** pour la gestion du backend, et une interface utilisateur en **JavaScript**.

## Technologies utilisées

- **Backend (Python)**
  - [Flask](https://flask.palletsprojects.com/)
  - [SQLite](https://www.sqlite.org/)

- **HTML/CSS**
- **Node.js**

## Installation

### Prérequis

- Python 3.x
- Node.js et npm
- Une base de données vectorielle.
<!-- SGBDR à déterminer -->

### Backend (Python/RASA)

1. Clonez le repository :
    ```bash
    git clone https://github.com/klueko/histerIA.git
    cd chatbot-histoire
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

<!-- 5. Configurez la base de données (SQLite/PostgreSQL) : -->
<!-- à faire -->

### Frontend (JavaScript)

1. Naviguez dans le dossier frontend :
    ```bash
    cd frontend
    ```

2. Installez les dépendances Node.js :
    ```bash
    npm install
    ```

3. Lancez l'interface utilisateur :
    ```bash
    npm start
    ```

    <!-- L'application sera accessible à `http://localhost:`. -->

