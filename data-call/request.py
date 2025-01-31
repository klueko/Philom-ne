import requests
import json

url = "https://datahub.bordeaux-metropole.fr//api/explore/v2.1/catalog/datasets/met_agenda/records"

response = requests.get(url)

if response.status_code == 200:
    data = response.json()

    with open("reponse.json", "w") as f:
        json.dump(data, f, indent=4)

    print("Réponse sauvegardée dans reponse.json")
else:
    print(f"Erreur {response.status_code}: {response.text}")
