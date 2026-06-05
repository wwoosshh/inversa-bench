"""E12 data prep: build matched (GSM8K-original = contaminated) vs (GSM-Symbolic = fresh) pairs.
GSM-Symbolic 'main' keeps difficulty fixed (same template, new numbers/names) and carries the
original GSM8K item inline, so acc(original) - acc(fresh) isolates the "this exact item was seen"
(contamination) + surface-fragility effect at constant difficulty. Gold numeric answers included.
Run: python scripts/gen_forward_gap_bank.py [N]
"""
import json
import re
import sys

from datasets import load_dataset


def goldnum(ans):
    m = re.search(r"####\s*([-\d,\.]+)", ans or "")
    return float(m.group(1).replace(",", "")) if m else None


N = int(sys.argv[1]) if len(sys.argv) > 1 else 80
s = load_dataset("apple/GSM-Symbolic", "main", split="test")

pairs, seen = [], set()
for it in s:
    if it["instance"] != 0 or it["original_id"] in seen:
        continue
    seen.add(it["original_id"])
    co, fr = goldnum(it["original_answer"]), goldnum(it["answer"])
    if co is None or fr is None:
        continue
    pairs.append({
        "original_id": it["original_id"],
        "contaminated_q": it["original_question"], "contaminated_ans": co,
        "fresh_q": it["question"], "fresh_ans": fr,
    })
pairs = pairs[:N]
json.dump({"name": "forward_gap_set",
           "note": "GSM8K-original (in training corpora = contaminated) vs GSM-Symbolic main "
                   "(fresh, same template/difficulty, canary-marked). Matched by template. "
                   "acc(contaminated) - acc(fresh) measures forward score inflation.",
           "pairs": pairs},
          open("data/banks/forward_gap_set.json", "w", encoding="utf-8"), indent=2)
print(f"wrote {len(pairs)} matched pairs")
print("example contaminated:", pairs[0]["contaminated_q"][:80], "->", pairs[0]["contaminated_ans"])
print("example fresh:       ", pairs[0]["fresh_q"][:80], "->", pairs[0]["fresh_ans"])
