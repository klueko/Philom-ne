import requests
import json

url = "https://datahub.bordeaux-metropole.fr/api/explore/v2.1/catalog/datasets/met_agenda/records"
all_data = []
offset = 0
limit = 100
max_records = 1500

while True:
    if len(all_data) >= max_records:
        print("Limite de 10 000 enregistrements atteinte.")
        break

    params = {
        'limit': limit,
        'offset': offset
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        print(json.dumps(data, indent=4, ensure_ascii=False))

        if 'results' in data:
            all_data.extend(data['results'])
        else:
            print("La clé 'results' n'est pas présente dans la réponse.")
            break

        if len(data['results']) < limit:
            break

        offset += limit
    else:
        print(f"Erreur {response.status_code}: {response.text}")
        break

all_data = all_data[:max_records]

with open("reponse.json", "w", encoding="utf-8") as f:
    json.dump(all_data, f, indent=4, ensure_ascii=False)

print(f"Réponse sauvegardée dans reponse.json avec {len(all_data)} enregistrements.")
