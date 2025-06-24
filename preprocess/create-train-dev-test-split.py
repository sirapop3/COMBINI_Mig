import json
import random


def split_data(json_file, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1):
    # Load JSON COMBINI-data
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Shuffle COMBINI-data
    random.shuffle(data)

    # Compute split sizes
    total = len(data)
    train_size = int(total * train_ratio)
    dev_size = int(total * dev_ratio)

    # Split COMBINI-data
    train_data = data[:train_size]
    dev_data = data[train_size:train_size + dev_size]
    test_data = data[train_size + dev_size:]

    # Save to separate files
    with open("COMBINI-data/train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2)

    with open("COMBINI-data/dev.json", "w", encoding="utf-8") as f:
        json.dump(dev_data, f, indent=2)

    with open("COMBINI-data/test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2)

    print("Data split completed: train.json, dev.json, test.json")


# Example usage
json_file = "COMBINI-data/pl-marker_formatted_adjudicated.json"  # Replace with your actual JSON file path
split_data(json_file)
