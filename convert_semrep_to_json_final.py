#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re, json, logging, xml.etree.ElementTree as ET
from collections import defaultdict, Counter
from difflib import SequenceMatcher
from tqdm import tqdm
import spacy

XML_IN   = "/projects/bdxz/sumnakkittikul/data/gs_308.semrep.out.xml"
ADJ_XML  = "/projects/bdxz/sumnakkittikul/data/adjudicated.xml"
MAP_IN   = "/projects/bdxz/sumnakkittikul/umls_semtypes.txt"
JSON_OUT = "/projects/bdxz/sumnakkittikul/data/converted_data.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S"
)

def load_semtype_map(path):
    mp = {}
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.strip() and not ln.startswith("#"):
                c, _, full = ln.strip().split("|", 2)
                mp[c] = full
    return mp

GREEK = {ord(k): v for k, v in {
    'α': 'a',  'β': 'b',  'γ': 'g',
    'Δ': 'D',  'δ': 'd',  'ε': 'e',
    'θ': 'th', 'κ': 'k',  'λ': 'l',
    'μ': 'm',  'ν': 'n',  'ξ': 'x',
    'π': 'p',  'ρ': 'r',  'σ': 's',
    'τ': 't',  'υ': 'y',  'φ': 'f',
    'χ': 'ch', 'ψ': 'ps', 'ω': 'w'
}.items()}

def canon(txt: str) -> str:
    txt = txt.translate(GREEK)
    txt = re.sub(r"['\"()\[\]{}]", " ", txt)
    txt = re.sub(r"[^0-9a-z+/]", " ", txt.lower())
    return re.sub(r"\s+", " ", txt).strip()

def find_token_indices(tokens, mention):
    want = canon(mention)
    if not want:
        return None, None
    want_sp = want.split()
    L = len(want_sp)
    tcanon = [canon(t) for t in tokens]
    for i in range(len(tcanon) - L + 1):
        if tcanon[i:i+L] == want_sp:
            return i, i+L-1
    for span in (0, 1, 2, 3):
        for i in range(max(0, len(tcanon) - L - span + 1)):
            cand = " ".join(tcanon[i:i+L+span])
            if abs(len(cand) - len(want)) > 3:
                continue
            if SequenceMatcher(None, cand, want).ratio() >= .85:
                return i, i+L+span-1
    return None, None

def build_trigger_maps(adj_path):
    trig_by_id   = defaultdict(list)
    trig_by_text = defaultdict(list)
    root = ET.parse(adj_path).getroot()
    for cit in root.iter("MedlineCitation"):
        pmid = cit.get("pmid", "").strip()
        for sent in cit.iter("Sentence"):
            sid  = sent.get("number", "").strip()
            sraw = sent.get("text", "")
            scon = canon(sraw)
            for pr in sent.findall(".//Predication"):
                if pr.get("inferred") == "true":
                    continue
                ptag  = pr.find("Predicate")
                if ptag is None:
                    continue
                rel   = (ptag.get("type") or "").upper()
                trig  = (ptag.get("text") or ptag.get("TEXT") or "").strip()
                if trig:
                    trig_by_id[(pmid, sid, rel)].append(trig)
                    trig_by_text[(pmid, scon, rel)].append(trig)
    logging.info("Loaded %d adjudicated trigger keys (by-id)", len(trig_by_id))
    return trig_by_id, trig_by_text

SEMTYPE_MAP               = load_semtype_map(MAP_IN)
TRIG_BY_ID, TRIG_BY_TEXT  = build_trigger_maps(ADJ_XML)
HIT_COUNTER               = Counter()
NLP = spacy.load("en_core_web_sm")

def convert(xml_in, json_out):
    root = ET.parse(xml_in).getroot()
    docs = root.findall(".//Document")

    with open(json_out, "w", encoding="utf-8") as fout:
        for doc in tqdm(docs, desc="Docs"):
            pmid = doc.get("id", "").lstrip("D")
            SENTS, NER, TRIGS, RELS, TRIPS = [], [], [], [], []
            ent_info, utters = {}, []
            for utt in doc.findall(".//Utterance"):
                txt  = utt.get("text", "")
                num  = utt.get("number", "").strip()
                ents, rels = [], []
                for ent in utt.findall(".//Entity"):
                    eid = ent.get("id")
                    ents.append(eid)
                    first = (ent.get("semtypes") or "").split(",")[0]
                    ent_info[eid] = {
                        "text": ent.get("text", ""),
                        "semtype": SEMTYPE_MAP.get(first, first)
                    }
                for pr in utt.findall(".//Predication"):
                    if pr.get("inferred") == "true":
                        continue
                    p = pr.find("Predicate"); s = pr.find("Subject"); o = pr.find("Object")
                    if None in (p, s, o):
                        continue
                    rels.append({
                        "type":    (p.get("type") or "").upper(),
                        "trg_txt": (p.get("text") or p.get("TEXT") or (p.text or "")).strip(),
                        "sub":     s.get("entityID"),
                        "obj":     o.get("entityID"),
                        "sub_rel": s.get("relSemType"),
                        "obj_rel": o.get("relSemType")
                    })
                utters.append(dict(text=txt, number=num, ents=ents, rels=rels))
            for utt in utters:
                doc_sp = NLP(utt["text"])
                toks, orig2new = [], {}
                for i, tok in enumerate(doc_sp):
                    if tok.text.strip():
                        orig2new[i] = len(toks)
                        toks.append(tok.text)
                SENTS.append(toks)
                ners, trig, rels, trip = [], [], [], []
                cache = {}
                for eid in utt["ents"]:
                    txt = ent_info[eid]["text"]
                    st  = ent_info[eid]["semtype"]
                    for r in utt["rels"]:
                        if eid == r["sub"] and r["sub_rel"]:
                            c = r["sub_rel"].split(",")[0]
                            st = SEMTYPE_MAP.get(c, c)
                        elif eid == r["obj"] and r["obj_rel"]:
                            c = r["obj_rel"].split(",")[0]
                            st = SEMTYPE_MAP.get(c, c)
                    if eid not in cache:
                        s, e = find_token_indices(toks, txt)
                        cache[eid] = (s, e)
                    s, e = cache[eid]
                    if s is not None:
                        ners.append([s, e, st])
                for r in utt["rels"]:
                    s0, s1 = cache.get(r["sub"], (None, None))
                    o0, o1 = cache.get(r["obj"], (None, None))
                    if None in (s0, o0):
                        continue
                    rels.append([s0, s1, o0, o1, r["type"]])
                    sid = utt["number"]; rel_label = r["type"]
                    key_id   = (pmid, sid, rel_label)
                    key_text = (pmid, canon(utt["text"]), rel_label)
                    if TRIG_BY_ID[key_id]:
                        surface = TRIG_BY_ID[key_id].pop(0)
                        HIT_COUNTER["id"] += 1
                    elif TRIG_BY_TEXT[key_text]:
                        surface = TRIG_BY_TEXT[key_text].pop(0)
                        HIT_COUNTER["text"] += 1
                    else:
                        surface = r["trg_txt"] or rel_label.replace("_", " ")
                        HIT_COUNTER["fallback"] += 1
                    t0, t1 = find_token_indices(toks, surface)
                    if t0 is None:
                        root_orig = next((t.i for t in doc_sp if t.dep_ == "ROOT"), None)
                        t0 = t1 = orig2new.get(root_orig, 0)
                    trig.append([t0, t1, surface])
                    trip.append([len(rels)-1, len(trig)-1])
                NER.append(sorted({tuple(x) for x in ners}))
                TRIGS.append(trig); RELS.append(rels); TRIPS.append(trip)
            fout.write(json.dumps(dict(
                doc_key   = pmid,
                sentences = SENTS,
                ner       = [list(map(list, n)) for n in NER],
                triggers  = TRIGS,
                relations = RELS,
                triplets  = TRIPS
            )) + "\n")
    logging.info("Stage-1 done → %s | hits(id=%d, text=%d, fallback=%d)",
                 json_out, HIT_COUNTER["id"], HIT_COUNTER["text"],
                 HIT_COUNTER["fallback"])

if __name__ == "__main__":
    convert(XML_IN, JSON_OUT)
