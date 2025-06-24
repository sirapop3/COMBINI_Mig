import json

#json_file = "COMBINI-data/pl-marker_formatted_adjudicated.json"
json_file = "COMBINI-data/train.json"
#json_file = "COMBINI-data/test.json"
#json_file = "COMBINI-data/dev.json"
# Read the original JSON file
with open(json_file, "r", encoding="utf-8") as file:
    data = json.load(file)

# Reformat each entry to remove the outer list structure
reformatted_data = []
for entry in data:
    reformatted_entry = {
        "doc_id": entry["doc_key"],
        "sentences": entry["sentences"],
        "ner": entry["ner"],
        "triggers": entry["triggers"],
        "relations": entry["relations"],
        "triplets": entry["triplets"]
    }
    reformatted_data.append(reformatted_entry)

# Write the reformatted JSON file
with open(json_file, "w", encoding="utf-8") as file:
    for entry in reformatted_data:
        json.dump(entry, file)
        file.write("\n")  # Write each object on a new line
