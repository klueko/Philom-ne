import json
import pandas as pd
import re

file_path = "reponse.json"

with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

def decode_json_fields(item):
    for field in ["attendancemode", "status"]:
        if field in item and isinstance(item[field], str):
            try:
                item[field] = json.loads(item[field])
            except json.JSONDecodeError:
                pass
    return item

def clean_string_field(field_value):
    if isinstance(field_value, str):
        cleaned_value = re.sub(r'<[^>]*>', '', field_value)
        cleaned_value = cleaned_value.replace("\\", "")
        cleaned_value = cleaned_value.replace("\\/", "/")
        cleaned_value = re.sub(r'&[a-zA-Z]+;', '', cleaned_value)
        cleaned_value = cleaned_value.replace('\"', '')
        cleaned_value = cleaned_value.replace("//", "/")
        return cleaned_value.strip()
    return field_value

def clean_labels(item):
    if isinstance(item, dict):
        cleaned_item = {k: v for k, v in item.items() if k not in ["attendancemode", "status"]}

        for field in ["location_website", "description_fr", "conditions_fr", "links"]:
            if field in item:
                cleaned_item[field] = clean_string_field(item[field])

        for field in ["attendancemode", "status"]:
            if field in item and isinstance(item[field], dict):
                cleaned_item[f"{field}.id"] = item[field].get("id")
                cleaned_item[f"{field}.label.fr"] = item[field].get("label", {}).get("fr")
        
        return cleaned_item
    return item

if isinstance(data, list):
    data = [decode_json_fields(item) for item in data]

data = [clean_labels(item) for item in data]

df = pd.json_normalize(data)

columns_to_drop = [
    "slug", "longdescription_fr", "originagenda_title", "updatedat", "firstdate_begin",
    "firstdate_end", "lastdate_begin", "lastdate_end", "location_coordinates",
    "location_insee", "location_tags", "originagenda_uid", "contributor_email",
    "contributor_contactnumber", "contributor_contactname",
    "contributor_contactposition", "contributor_organization", "location_coordinates.lon",
    "location_coordinates.lat"
]

df_cleaned = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

output_path = "reponse_cleaned.json"
df_cleaned.to_json(output_path, orient="records", indent=4, force_ascii=False)

print(f"Fichier nettoyé sauvegardé sous : {output_path}")
