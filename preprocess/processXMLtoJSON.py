import xml.etree.ElementTree as ET
import json
import requests
import string
import re

import spacy
nlp = spacy.load("en_core_web_sm")


def fetch_pubtator_data(pmid):
    url = f"https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/biocjson?pmids={pmid}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        title, abstract = "", ""
        for doc in data.get('PubTator3', []):
            for passage in doc.get('passages', []):
                passage_type = passage.get('infons', {}).get('type')
                if passage_type == "title":
                    title = passage.get('text', "")
                elif passage_type == "abstract":
                    abstract = passage.get('text', "")
        return title, abstract
    return "", ""


def strip_punctuation(text):
    return text.strip(string.punctuation)

def find_token_indices(entity_text, full_text_tokens):
    """Find the start and end indices of an entity in tokenized text, handling mismatches."""
    entity_tokens = entity_text.split()
    entity_tokens = [token.strip(string.punctuation).lower() for token in entity_tokens]
    full_text_tokens = [token.strip(string.punctuation).lower() for token in full_text_tokens]

    for i in range(len(full_text_tokens) - len(entity_tokens) + 1):
        if full_text_tokens[i:i + len(entity_tokens)] == entity_tokens:
            return i, i + len(entity_tokens) - 1

    return None, None  # Return None if not found


def normalize_text(text):
    """Use spaCy for better tokenization and punctuation handling."""
    text = text.lower()  # Case normalization
    # Special case: Convert '(+)' to '+'
    text = text.replace("(+)", "+")
    tokens = [token.text for token in nlp(text) if not token.is_punct]  # Remove punctuations
    return " ".join(tokens)

def verify_entity_text_match(xml_entities, token_entities, full_text, sentences):
    for (expected_text), (token_start, token_end) in zip(xml_entities, token_entities):
        if token_start is None or token_end is None:
            print(f"Skipping verification: Could not find entity '{expected_text}' in tokenized text.")
            continue  # Skip this entity if it wasn't found

        extracted_text_xml = expected_text
        extracted_text_tokens = " ".join(full_text.split()[token_start:token_end + 1])

        # Debugging output
        #print(f"\nOriginal Extracted from XML: '{extracted_text_xml}'")
        #print(f"Original Extracted from Tokens: '{extracted_text_tokens}'")

        # Normalize both texts before comparison
        extracted_text_xml = normalize_text(extracted_text_xml)
        extracted_text_tokens = normalize_text(extracted_text_tokens)

        #print(f"Normalized Extracted from XML: '{extracted_text_xml}'")
        #print(f"Normalized Extracted from Tokens: '{extracted_text_tokens}'")

        assert extracted_text_xml == extracted_text_tokens, f"Entity text mismatch: Expected '{extracted_text_xml}', but got '{extracted_text_tokens}'"

    print("Entity text verification passed.")


def parse_xml_to_json(xml_file, json_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    data = []

    for citation in root.findall("MedlineCitation"):
        pmid = citation.get("pmid")
        doc_key = f"{pmid}"

        title, abstract = fetch_pubtator_data(pmid)
        full_text = title + " " + abstract

        xml_sentences = []
        sentences = []
        ner = []
        relations = []
        triggers = []
        triplets = []

        xml_entities = []
        token_entities = []

        for sentence in citation.findall("Sentence"):
            text = sentence.get("text")
            xml_sentences.append(text.split())
            ner.append([])
            relations.append([])
            triggers.append([])
            triplets.append([])

            predications = sentence.find("Predications")
            if predications is not None:
                for predication in predications.findall("Predication"):
                    predicate = predication.find("Predicate")
                    subject = predication.find("Subject")
                    obj = predication.find("Object")

                    if predicate is not None and subject is not None and obj is not None:
                        predicate_type = predicate.get("type")
                        subj_text = subject.get("text")
                        obj_text = obj.get("text")

                        xml_entities.append((subj_text))
                        xml_entities.append((obj_text))

                        full_text_tokens = full_text.split()
                        # Use this function instead of .index()
                        subj_token_start, subj_token_end = find_token_indices(subj_text, full_text_tokens)
                        obj_token_start, obj_token_end = find_token_indices(obj_text, full_text_tokens)

                        if subj_token_start is None or obj_token_start is None:
                            print(f"Warning: Could not find entity '{subj_text}' or '{obj_text}' in tokenized text.")

                        token_entities.append((subj_token_start, subj_token_end))
                        token_entities.append((obj_token_start, obj_token_end))

                        try:
                            ner[-1].append([subj_token_start, subj_token_end, subject.find("RelationSemanticType").text])
                            ner[-1].append([obj_token_start, obj_token_end, obj.find("RelationSemanticType").text])
                            relations[-1].append(
                                [subj_token_start, subj_token_end, obj_token_start, obj_token_end, predicate_type, ""])
                        except Exception as e:
                            print(f"Error processing entity positions: {e}")

        for sentence in (title + ". " + abstract).split(". "):
            tokens = []
            for token in sentence.split(" "):
                tokens.append(token)
            sentences.append(tokens)
            ner.append([])
            triggers.append([])
            relations.append([])
            triplets.append([])

        verify_entity_text_match(xml_entities, token_entities, full_text, sentences)

        data.append({
            "doc_key": doc_key,
            "sentences": sentences,
            "ner": ner,
            "triggers": triggers,
            "relations": relations,
            "triplets": triplets
        })

    with open(json_file, "w") as f:
        json.dump(data, f, indent=2)


# Example usage:
# parse_xml_to_json("example.xml", "output.json")


parse_xml_to_json("../COMBINI-data/adjudicated.xml", "COMBINI-data/pl-marker_formatted_adjudicated.json")

