import pandas as pd
import json

file_path = "reponse.json"
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

if isinstance(data, list):
    df = pd.json_normalize(data)
else:
    print("Le format de données n'est pas une liste.")
    exit()

columns_to_drop = [
    "slug", "description_fr", "originagenda_title", "updatedat", "firstdate_begin",
    "firstdate_end", "lastdate_begin", "lastdate_end", "location_coordinates",
    "location_insee", "location_tags", "originagenda_uid", "contributor_email",
    "contributor_contactnumber", "contributor_contactname",
    "contributor_contactposition", "contributor_organization", "location_coordinates.lon",
        "location_coordinates.lat"
]

df_cleaned = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

output_path = "reponse_nettoyee.json"
df_cleaned.to_json(output_path, orient="records", indent=4, force_ascii=False)

print(f"Fichier nettoyé sauvegardé sous : {output_path}")
