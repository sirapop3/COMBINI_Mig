#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, argparse, sys
from pathlib import Path
from collections import Counter

def span_ok(tokens, s, e):
    if any(t is None for t in (s, e)):
        return False
    if not (0 <= s <= e < len(tokens)):
        return False
    return bool(" ".join(tokens[s:e+1]).strip())

def main(fp: Path):
    problems = Counter()

    with fp.open(encoding="utf-8") as f:
        for ln_no, line in enumerate(f, 1):
            data = json.loads(line)
            doc = data["doc_key"]

            for sid, sent in enumerate(data["sentences"]):
                for s, e, _ in data["ner"][sid]:
                    if not span_ok(sent, s, e):
                        problems["NER"] += 1
                        print(f"[NER] {doc} s{sid}: {s}-{e} bad")

                for s, e, _ in data["triggers"][sid]:
                    if not span_ok(sent, s, e):
                        problems["TRIG"] += 1
                        print(f"[TRIG] {doc} s{sid}: {s}-{e} bad")

                for idx, (s0, e0, s1, e1, _) in enumerate(data["relations"][sid]):
                    if not span_ok(sent, s0, e0):
                        problems["REL_SUBJ"] += 1
                        print(f"[REL ] {doc} s{sid}: rel#{idx} subj {s0}-{e0} bad")
                    if not span_ok(sent, s1, e1):
                        problems["REL_OBJ"] += 1
                        print(f"[REL ] {doc} s{sid}: rel#{idx} obj  {s1}-{e1} bad")

                num_rel = len(data["relations"][sid])
                num_trg = len(data["triggers"][sid])
                for r_idx, t_idx in data["triplets"][sid]:
                    if not (0 <= r_idx < num_rel):
                        problems["TRIP_REL"] += 1
                        print(f"[TRIP] {doc} s{sid}: relation index {r_idx} OOB")
                    if not (0 <= t_idx < num_trg):
                        problems["TRIP_TRG"] += 1
                        print(f"[TRIP] {doc} s{sid}: trigger index {t_idx} OOB")

    if problems:
        print("\nSummary of issues:")
        for k, v in problems.items():
            print(f"  {k:<9}: {v}")
        sys.exit(1)
    else:
        print("✓ All span indices and triplet references are valid.")
        sys.exit(0)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Validate Stage-2 PL-Marker JSON.")
    p.add_argument(
        "-f", "--file", default="/projects/bdxz/sumnakkittikul/data/converted_data_final.json",
        help="path to the Stage-2 JSON lines file (default: converted_data_final.json)"
    )
    args = p.parse_args()
    main(Path(args.file))
