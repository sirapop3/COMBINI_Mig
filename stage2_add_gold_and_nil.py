#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, logging, pandas as pd
from copy import deepcopy
from difflib import SequenceMatcher

JSON_IN   = "/projects/bdxz/sumnakkittikul/data/converted_data.json"
CSV_MISS  = "missing_in_model.csv"
CSV_FP    = "false_positives.csv"
CSV_MISM  = "label_mismatches.csv"
JSON_OUT  = "/projects/bdxz/sumnakkittikul/data/converted_data_final.json"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

def canon(t: str) -> str:
    t = t.lower()
    t = "".join(ch for ch in t if ch.isalnum() or ch.isspace())
    return " ".join(t.split())

def tok_span(tokens, start, end):
    return canon(" ".join(tokens[start:end+1]))

def span_match(a, b):
    return SequenceMatcher(None, a, b).ratio() >= .85

miss_df = pd.read_csv(CSV_MISS)
mism_df = pd.read_csv(CSV_MISM)
fp_df   = pd.read_csv(CSV_FP)

gold_df = pd.concat([miss_df, mism_df], ignore_index=True)

gold_df["gkey"] = gold_df.apply(
    lambda r: (str(r.docid), int(r.sentenceid_gold),
               canon(r.subject_text_gold), canon(r.object_text_gold)), axis=1)
gold_map = {k: i for i, k in gold_df.gkey.items()}

fp_df["fkey"] = fp_df.apply(
    lambda r: (str(r.docid), int(r.sentenceid_model),
               canon(r.subject_text_model), canon(r.object_text_model)), axis=1)
fp_set = set(fp_df.fkey)

with open(JSON_OUT, "w", encoding="utf-8") as fout, \
     open(JSON_IN,  encoding="utf-8") as fin:

    for ln in fin:
        data = json.loads(ln)
        doc = data["doc_key"]

        for s_id, sent in enumerate(data["sentences"]):
            rel_map = {}
            for idx, (s0,s1,o0,o1,label) in enumerate(data["relations"][s_id]):
                skey = tok_span(sent,s0,s1)
                okey = tok_span(sent,o0,o1)
                rel_map[(skey,okey)] = idx

            for (skey,okey), idx in rel_map.items():
                key = (doc, s_id, skey, okey)
                if key in fp_set:
                    data["relations"][s_id][idx][4] = "NIL"
                    data["triplets"][s_id] = [tr for tr in data["triplets"][s_id]
                                              if tr[0] != idx]
                    continue
                if key in gold_map:
                    gold_lbl = gold_df.loc[gold_map.pop(key),"relation_gold"]
                    data["relations"][s_id][idx][4] = gold_lbl

            for key in list(gold_map):
                g_doc, g_sid, g_sub, g_obj = key
                if g_doc != doc or g_sid != s_id:
                    continue
                row = gold_df.loc[gold_map.pop(key)]
                try:
                    s0,s1 = next((i,j) for i in range(len(sent))
                                 for j in range(i,len(sent))
                                 if span_match(tok_span(sent,i,j), g_sub))
                    o0,o1 = next((i,j) for i in range(len(sent))
                                 for j in range(i,len(sent))
                                 if span_match(tok_span(sent,i,j), g_obj))
                except StopIteration:
                    logging.warning("[ADD-WARN] %s s%d: cannot locate %s / %s",
                                    doc, s_id, row.subject_text_gold, row.object_text_gold)
                    continue
                new_rel_idx = len(data["relations"][s_id])
                data["relations"][s_id].append([s0,s1,o0,o1,row.relation_gold])
                trigger_word = row.relation_gold.replace("_"," ")
                try:
                    t0,t1 = next((i,i) for i,t in enumerate(sent)
                                 if span_match(canon(t), canon(trigger_word)))
                except StopIteration:
                    t0 = t1 = 0
                new_trg_idx = len(data["triggers"][s_id])
                data["triggers"][s_id].append([t0,t1,row.relation_gold])
                data["triplets"][s_id].append([new_rel_idx, new_trg_idx])

        fout.write(json.dumps(data) + "\n")

missing = len(gold_map)
logging.info("Stage-2 done → %s  (unmatched adjudicated relations: %d)", JSON_OUT, missing)
