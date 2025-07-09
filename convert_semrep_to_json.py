import os
import xml.etree.ElementTree as ET
import json
import spacy
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_semtype_map(mapping_path):
    semtype_map = {}
    with open(mapping_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('|', 2)
            if len(parts) == 3:
                code, _, fullname = parts
                semtype_map[code] = fullname
    return semtype_map

SEMTYPE_MAP = load_semtype_map('umls_semtypes.txt')

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logging.error("Spacy model 'en_core_web_sm' not found.")
    logging.error("Please run: python -m spacy download en_core_web_sm")
    exit()

def find_token_indices(sentence_tokens, entity_text):
    entity_tokens = [tok.text.lower() for tok in nlp(entity_text)]
    sent_lower = [tok.lower() for tok in sentence_tokens]
    for i in range(len(sent_lower) - len(entity_tokens) + 1):
        if sent_lower[i:i + len(entity_tokens)] == entity_tokens:
            return i, i + len(entity_tokens) - 1
    entity_simple = "".join(entity_tokens)
    for i in range(len(sent_lower) - len(entity_tokens) + 1):
        window = "".join(sent_lower[i:i + len(entity_tokens)])
        if window == entity_simple:
            return i, i + len(entity_tokens) - 1
    return None, None

def convert_semrep_to_json(xml_path, json_path, semtype_map):
    logging.info(f"Converting '{xml_path}' → '{json_path}'…")
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        logging.error(f"XML parse error: {e}")
        return

    documents = root.findall('.//Document')
    if not documents:
        logging.warning("No <Document> elements found.")
        return

    with open(json_path, 'w', encoding='utf-8') as fout:
        for doc in tqdm(documents, desc="Documents"):
            raw_id = doc.get('id', '')
            doc_key = raw_id[1:] if raw_id.startswith('D') else raw_id
            entity_map = {}
            utterances = []
            for utt in doc.findall('.//Utterance'):
                text = utt.get('text', '')
                uid  = utt.get('id', '')
                for ent in utt.findall('.//Entity'):
                    eid   = ent.get('id')
                    txt   = ent.get('text', '')
                    raw_types = ent.get('semtypes', '')
                    first_code = raw_types.split(',')[0] if raw_types else None
                    full_type  = semtype_map.get(first_code, first_code)
                    entity_map[eid] = {'text': txt, 'semtype': full_type}
                rels = []
                for pred in utt.findall('.//Predication'):
                    if pred.get('inferred') == 'true':
                        continue
                    p = pred.find('Predicate')
                    s = pred.find('Subject')
                    o = pred.find('Object')
                    if p is None or s is None or o is None:
                        continue
                    rels.append({
                        'type':    p.get('type'),
                        'subj':    s.get('entityID'),
                        'obj':     o.get('entityID')
                    })
                utterances.append({'id': uid, 'text': text, 'relations': rels})

            sentences = []
            all_ners   = []
            all_rels   = []
            global_offset = 0

            for utt in utterances:
                sent_text = utt['text']
                doc_spacy = nlp(sent_text)
                toks = [t.text for t in doc_spacy if t.text.strip()]
                sentences.append(toks)
                ners  = []
                rels  = []
                cache = {}
                base  = global_offset
                for rel in utt['relations']:
                    sid, oid = rel['subj'], rel['obj']
                    if sid not in entity_map or oid not in entity_map:
                        logging.warning(f"{doc_key}: missing entity {sid} or {oid}")
                        continue
                    if sid not in cache:
                        s0, s1 = find_token_indices(toks, entity_map[sid]['text'])
                        cache[sid] = (s0, s1)
                    s0, s1 = cache[sid]
                    if oid not in cache:
                        o0, o1 = find_token_indices(toks, entity_map[oid]['text'])
                        cache[oid] = (o0, o1)
                    o0, o1 = cache[oid]
                    if s0 is None or o0 is None:
                        logging.warning(
                            f"{doc_key}: couldn’t locate '{entity_map[sid]['text']}' "
                            f"or '{entity_map[oid]['text']}' in tokens"
                        )
                        continue
                    gs, ge = s0 + base, s1 + base
                    os_, oe = o0 + base, o1 + base
                    ners.append([gs, ge, entity_map[sid]['semtype']])
                    ners.append([os_, oe, entity_map[oid]['semtype']])
                    rels.append([gs, ge, os_, oe, rel['type']])
                uniq_ners = [list(x) for x in set(tuple(n) for n in ners)]
                all_ners.append(sorted(uniq_ners))
                all_rels.append(sorted(rels))
                global_offset += len(toks)

            out = {
                "doc_key":   doc_key,
                "sentences": sentences,
                "ner":       all_ners,
                "triggers":  [[] for _ in sentences],
                "relations": all_rels,
                "triplets":  [[] for _ in sentences]
            }
            fout.write(json.dumps(out) + "\n")

    logging.info("Conversion complete.")

if __name__ == "__main__":
    XML_IN   = "/projects/bdxz/sumnakkittikul/data/gs_308.semrep.out.xml"
    MAPPING  = "/projects/bdxz/sumnakkittikul/umls_semtypes.txt"
    JSON_OUT = "/projects/bdxz/sumnakkittikul/data/converted_data.json"
    if not os.path.exists(MAPPING):
        logging.error(f"Semtype mapping file not found: {MAPPING}")
        exit(1)
    convert_semrep_to_json(XML_IN, JSON_OUT, SEMTYPE_MAP)
