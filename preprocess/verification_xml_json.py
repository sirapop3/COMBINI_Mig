import spacy
import xml.etree.ElementTree as ET
import json

# Load spaCy model for text normalization
nlp = spacy.load("en_core_web_sm")


# Assuming you have already defined the count_relations_in_xml and count_relations_in_json functions

def extract_sentence_from_xml(xml_file, idx):
    """
    Extract the sentence from the XML based on the given index.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    sentences = root.findall(".//Sentence")  # Adjust this based on your XML structure
    if idx < len(sentences):
        return sentences[idx].get("text")  # Or use the appropriate XML tag/path
    return ""


def extract_sentence_from_json(json_file, idx):
    """
    Extract the sentence from the JSON based on the given index.
    """
    with open(json_file, "r") as file:
        data = json.load(file)
        sentences = data[idx]['sentences']  # Assuming 'sentences' is a list in your JSON structure
        if idx < len(sentences):
            return " ".join(sentences[idx])  # Join tokenized words back into a sentence
    return ""


def normalize_text(text):
    """
    Normalize text using spaCy: convert to lowercase and remove unnecessary punctuation/whitespace.
    """
    doc = nlp(text)
    normalized_text = " ".join([token.text.lower() for token in doc if not token.is_punct])
    return normalized_text


def get_entity_text_from_xml(root, doc_idx):
    """
    Extract both subject and object texts from the XML for a given sentence.
    The entity is expected to be in the form of [start_char, end_char, entity_type].
    """
    # Get the corresponding Sentence for the current doc_idx
    sentence = root.findall(f".//MedlineCitation/Sentence")[doc_idx]

    # Initialize lists for subject and object texts
    subjects_text = []
    objects_text = []

    # Find all Predications (they contain Subjects and Objects)
    for predication in sentence.findall(".//Predications/Predication"):
        # Extract Subject and Object texts
        subject = predication.find("Subject")
        if subject is not None:
            subjects_text.append(subject.get("text"))

        obj = predication.find("Object")
        if obj is not None:
            objects_text.append(obj.get("text"))

    return subjects_text, objects_text


def verify_token_matching(xml_file, json_file):
    """
    Verify that the tokens in the JSON file correctly match the entity text in the XML file.
    The entity should be found in either the Subject or Object in the XML for the corresponding sentence.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    with open(json_file, "r") as file:
        data = json.load(file)

    for idx, doc in enumerate(data):
        # Concatenate all sentences in the document to create a long list of tokens
        all_tokens = [token for sentence in doc['sentences'] for token in sentence]
        print('abstract', all_tokens)# Flatten list of lists

        # Get subject and object texts from XML
        subjects_text, objects_text = get_entity_text_from_xml(root, idx)

        # Normalize the texts from XML (subjects and objects)
        normalized_subjects = [normalize_text(text) for text in subjects_text]
        normalized_objects = [normalize_text(text) for text in objects_text]

        for ner_entity in doc['ner']:
            # Check if ner_entity is empty (skip it if it is)
            if not ner_entity:
                continue

            print(f"Inspecting ner_entity: {ner_entity}")

            # Extract token start and end indices from each sublist in ner_entity
            for entity in ner_entity:
                if isinstance(entity, list) and len(
                        entity) == 3:  # Ensure the list has at least 2 elements (start and end)
                    token_start, token_end = int(entity[0]), int(entity[1])  # Ensure they are integers

                    # Adjust token_start and token_end to be relative to the concatenated list of tokens
                    extracted_tokens = all_tokens[token_start:token_end + 1]
                    print('extracted_tokens', extracted_tokens)
                    extracted_text_json = " ".join(extracted_tokens)
                    print('extracted_text_json', extracted_text_json)

                    # Normalize the extracted text from JSON
                    normalized_extracted_json = normalize_text(extracted_text_json)

                    # Check if normalized extracted text matches either subject or object text
                    if normalized_extracted_json not in normalized_subjects and \
                            normalized_extracted_json not in normalized_objects:
                        raise AssertionError(
                            f"Entity text mismatch for entity {ner_entity}: Expected one of {normalized_subjects} or {normalized_objects}, but got '{normalized_extracted_json}'"
                        )

    print("Token matching verification passed.")


def verify_relation_counts(xml_file, json_file):
    """
    Verify that the number of relations in each sentence matches between the XML and JSON files.
    Also prints out sentences with relations in XML but no corresponding relations in JSON.
    """
    xml_relation_counts = count_relations_in_xml(xml_file)
    json_relation_counts = count_relations_in_json(json_file)

    # Initialize lists to hold sentences with relations from XML and JSON
    xml_idx = 0
    for xml_count in xml_relation_counts:
        if xml_count > 0:
            xml_sentence = extract_sentence_from_xml(xml_file, xml_idx)
            xml_relation_sentences.append((xml_sentence, xml_count))
        xml_idx += 1

    # Collect sentences with relations from the JSON
    json_idx = 0
    for json_count in json_relation_counts:
        if json_count > 0:
            json_sentence = extract_sentence_from_json(json_file, json_idx)
            json_relation_sentences.append((json_sentence, json_count))
        json_idx += 1

    # Now we compare the relations between XML and JSON
    missing_relations = [
        sentence for sentence in xml_relation_sentences if sentence not in json_relation_sentences
    ]

    if missing_relations:
        print("\nXML sentences with relations but no matching relations in JSON:")
        for sentence, count in missing_relations:
            print(f"Sentence: {sentence} | Relation count: {count}")

    # Now verify that the number of relations match for each sentence
    if len(xml_relation_counts) != len(json_relation_counts):
        raise AssertionError(
            f"Sentence count mismatch for relations: XML has {len(xml_relation_counts)} sentences, JSON has {len(json_relation_counts)}."
        )

    # Compare the counts of relations in XML and JSON (only non-empty relations)
    for i, (xml_count, json_count) in enumerate(zip(xml_relation_counts, json_relation_counts)):
        if xml_count != json_count:
            raise AssertionError(
                f"Relation count mismatch in sentence {i + 1}: Expected {xml_count}, but got {json_count} in JSON."
            )

    print("Relation count verification passed. All sentences have matching relation counts.")


def count_relations_in_xml(xml_file):
    """
    Count the number of relations in the XML file. Return a list where each item
    corresponds to the number of relations in a sentence.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    relation_counts = []

    for sentence in root.findall(".//Sentence"):
        relations = sentence.findall(".//Relation")  # Adjust according to your XML structure
        relation_counts.append(len(relations))

    return relation_counts


def count_relations_in_json(json_file):
    """
    Count the number of relations in the JSON file. Return a list where each item
    corresponds to the number of relations in a sentence.
    """
    with open(json_file, "r") as file:
        data = json.load(file)

    relation_counts = []
    for doc in data:
        relations = doc.get("relations", [])
        non_empty_relations = [r for r in relations if r]  # Filter out empty relations
        relation_counts.append(len(non_empty_relations))

    return relation_counts


xml_filepath = "../COMBINI-data/adjudicated.xml"  # Replace with the actual XML file path
json_filepath = "../COMBINI-data/pl-marker_formatted_adjudicated.json"  # Replace with the actual JSON file path

print("Starting verification...\n")
verify_token_matching(xml_filepath, json_filepath)
verify_relation_counts(xml_filepath, json_filepath)
print("\nAll verifications passed successfully!")
