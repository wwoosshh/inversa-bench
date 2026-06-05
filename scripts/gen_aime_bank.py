"""E10 data prep: AIME 2024 + 2025 — a HARD, non-saturated forward benchmark with integer
answers (0-999, clean exact-match grading). Used to compare the *discrimination power* of a hard
forward benchmark vs IGS on the same models. Run: python scripts/gen_aime_bank.py
"""
import json

from datasets import load_dataset


def norm(a):
    try:
        return float(str(a).strip())
    except Exception:
        return None


probs = []
for it in load_dataset("Maxwell-Jia/AIME_2024")["train"]:
    a = norm(it["Answer"])
    if a is not None:
        probs.append({"problem": it["Problem"], "answer": a, "year": 2024})
for it in load_dataset("yentinglin/aime_2025")["train"]:
    a = norm(it["answer"])
    if a is not None:
        probs.append({"problem": it["problem"], "answer": a, "year": 2025})

json.dump({"name": "aime_set",
           "note": "AIME 2024+2025, integer answers; hard non-saturated forward benchmark "
                   "for the E10 discrimination-power comparison vs IGS.",
           "problems": probs},
          open("data/banks/aime_set.json", "w", encoding="utf-8"), indent=2)
print(f"wrote {len(probs)} AIME problems")
