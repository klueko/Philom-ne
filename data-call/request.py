import requests
import json

url = "https://datahub.bordeaux-metropole.fr//api/explore/v2.1/catalog/datasets/met_agenda/records"

response = requests.get(url)

if response.status_code == 200:
    data = response.json()

    with open("reponse.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print("Réponse sauvegardée dans reponse.json")
else:
    print(f"Erreur {response.status_code}: {response.text}")
