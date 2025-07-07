import xml.etree.ElementTree as ET
import json
import spacy
import logging
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")


def find_token_indices(sentence_tokens, entity_text):
    # from processXMLtoJSON
    entity_tokens = [token.text.lower() for token in nlp(entity_text)]
    sentence_tokens_lower = [token.lower() for token in sentence_tokens]

    for i in range(len(sentence_tokens_lower) - len(entity_tokens) + 1):
        if sentence_tokens_lower[i:i + len(entity_tokens)] == entity_tokens:
            return i, i + len(entity_tokens) - 1
            
    # Fallback for cases where tokenization differs significantly
    # This is a simple fallback and might not cover all edge cases.
    entity_text_simple = "".join(entity_tokens)
    for i in range(len(sentence_tokens_lower) - len(entity_tokens) + 1):
        window_simple = "".join(sentence_tokens_lower[i:i + len(entity_tokens)])
        if window_simple == entity_text_simple:
            return i, i + len(entity_tokens) - 1

    return None, None


def convert_semrep_to_json(xml_path, json_path):
    print("start conversion")
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        logging.error(f"Failed to parse XML file: {e}")
        return

    with open(json_path, 'w', encoding='utf-8') as f_out:
        # find all document tags
        documents = root.findall('.//Document')
        if not documents:
            logging.warning("No <Document> tags found in the XML file.")
            return

        for doc in tqdm(documents, desc="Processing Documents"):
            doc_id_raw = doc.get('id')
            doc_key = doc_id_raw[1:] if doc_id_raw.startswith('D') else doc_id_raw # dont include "D" at the beginning
            
            #  first step: gather all data from the document
            entity_map = {}
            utterances_data = []

            for utterance in doc.findall('.//Utterance'):
                sent_text = utterance.get('text')
                sent_id = utterance.get('id')
                
                # store entities in a map for easy lookup
                for entity in utterance.findall('.//Entity'):
                    entity_id = entity.get('id')
                    entity_map[entity_id] = {
                        'text': entity.get('text'),
                        'type': entity.get('name'), # using 'name'
                    }

                # store relations associated with this utterance
                relations_in_sent = []
                for predication in utterance.findall('.//Predication'):
                    # skip inferred relations
                    if predication.get('inferred') == 'true':
                        continue

                    predicate = predication.find('Predicate')
                    subject = predication.find('Subject')
                    obj = predication.find('Object')

                    if predicate is not None and subject is not None and obj is not None:
                        relations_in_sent.append({
                            'rel_type': predicate.get('type'),
                            'subj_id': subject.get('entityID'),
                            'obj_id': obj.get('entityID')
                        })
                
                utterances_data.append({
                    'id': sent_id,
                    'text': sent_text,
                    'relations': relations_in_sent
                })

            # second step: process gathered data into the target JSON format
            doc_sentences = []
            doc_ners = []
            doc_relations = []

            for utterance_info in utterances_data:
                sent_text = utterance_info['text']
                
                # tokenize sentence using spacy for better accuracy
                spacy_doc = nlp(sent_text)
                sentence_tokens = [token.text for token in spacy_doc]
                doc_sentences.append(sentence_tokens)

                sent_ners = []
                sent_relations = []
                
                # cache to avoid re-calculating token indices for the same entity
                token_indices_cache = {}

                for rel_info in utterance_info['relations']:
                    subj_id = rel_info['subj_id']
                    obj_id = rel_info['obj_id']

                    if subj_id not in entity_map or obj_id not in entity_map:
                        logging.warning(f"In {doc_key}, skipping relation with missing entity: {subj_id} or {obj_id}")
                        continue

                    # gettgin subject info
                    if subj_id not in token_indices_cache:
                        subj_text = entity_map[subj_id]['text']
                        start, end = find_token_indices(sentence_tokens, subj_text)
                        token_indices_cache[subj_id] = (start, end)
                    subj_start_tok, subj_end_tok = token_indices_cache[subj_id]
                    
                    if obj_id not in token_indices_cache:
                        obj_text = entity_map[obj_id]['text']
                        start, end = find_token_indices(sentence_tokens, obj_text)
                        token_indices_cache[obj_id] = (start, end)
                    obj_start_tok, obj_end_tok = token_indices_cache[obj_id]

                    # skip if either entity could not be located
                    if subj_start_tok is None or obj_start_tok is None:
                        logging.warning(f"In {doc_key}, could not locate entities for relation. Subject: '{entity_map[subj_id]['text']}', Object: '{entity_map[obj_id]['text']}'")
                        continue
                    
                    # add ner entries for both subject and object
                    sent_ners.append([subj_start_tok, subj_end_tok, entity_map[subj_id]['type']])
                    sent_ners.append([obj_start_tok, obj_end_tok, entity_map[obj_id]['type']])
                    
                    # ddd the relation entry
                    sent_relations.append([
                        subj_start_tok, subj_end_tok,
                        obj_start_tok, obj_end_tok,
                        rel_info['rel_type'],
                        ""
                    ])

                # remove duplicate ner entries that might have been added
                unique_ners = [list(t) for t in set(tuple(item) for item in sent_ners)]
                
                doc_ners.append(sorted(unique_ners))
                doc_relations.append(sorted(sent_relations))

            # third step: assemble and write the final JSON object
            final_doc_obj = {
                "doc_key": doc_key,
                "sentences": doc_sentences,
                "ner": doc_ners,
                "relations": doc_relations,
                "triggers": [[] for _ in doc_sentences], # emptty
                "triplets": [[] for _ in doc_sentences]  # also empty
            }
            
            f_out.write(json.dumps(final_doc_obj) + '\n')

    logging.info(f"Conversion done. Output saved to '{json_path}'.")


if __name__ == '__main__':
    input_xml_file = '/projects/bdxz/sumnakkittikul/data/gs_308.semrep.out.xml'
    output_json_file = '/projects/bdxz/sumnakkittikul/data/converted_data.json'
    
    convert_semrep_to_json(input_xml_file, output_json_file)