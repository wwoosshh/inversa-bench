"""Select ~100 measurement-worthy models from the curated candidates for the IGS leaderboard.
Rules: drop dated-snapshot duplicates when a base sibling exists; cap big families; keep tier
spread (weak->frontier). Writes data/banks/leaderboard_roster.json. Run: python scripts/select_roster.py
"""
import collections
import json
import re

ids = json.load(open("data/results/_leaderboard_candidates.json"))
fam = collections.defaultdict(list)
for i in ids:
    fam[i.split("/")[0]].append(i)

DATED = re.compile(r"-(\d{4}(-\d\d-\d\d)?|20\d{6}|\d{2}-\d{2})$")

def dedup_snapshots(items):
    stems = set()
    base = []
    # keep non-dated first; record their stem so dated siblings can be dropped
    for i in sorted(items, key=lambda x: (bool(DATED.search(x)), x)):
        stem = DATED.sub("", i)
        if DATED.search(i) and stem in stems:
            continue  # dated dup of an existing base
        stems.add(stem)
        base.append(i)
    return base

# flagship/keep priority + ensure a weak anchor per big family
PRIO = ("max", "pro", "plus", "opus", "large", "ultra", "premier", "405b", "235b", "120b",
        "a22b", "maverick", "thinking", "v3.2", "v4", "r1", "5", "4.6", "4.7", "4.8")
CAP = {"openai": 13, "qwen": 12, "anthropic": 10, "google": 9, "mistralai": 8}

def score(i):
    s = sum(2 for k in PRIO if k in i)
    return s - 0.01 * len(i)

roster = []
for f, items in fam.items():
    items = dedup_snapshots(items)
    cap = CAP.get(f, 6)
    if len(items) > cap:
        # keep top-`cap-1` by flagship score + 1 smallest (weak anchor for tier spread)
        ranked = sorted(items, key=score, reverse=True)
        small = min(items, key=lambda x: (("7b" not in x and "8b" not in x and "mini" not in x
                                           and "lite" not in x and "small" not in x and "air" not in x), len(x)))
        keep = ranked[:cap - 1]
        if small not in keep:
            keep = keep[:cap - 1] + [small]
        items = keep
    roster.extend(items)

roster = sorted(set(roster))
json.dump(roster, open("data/banks/leaderboard_roster.json", "w"), indent=1)
print(f"selected {len(roster)} models across {len(set(r.split('/')[0] for r in roster))} families\n")
byf = collections.defaultdict(list)
for r in roster:
    byf[r.split("/")[0]].append(r.split("/", 1)[1])
for f in sorted(byf, key=lambda k: -len(byf[k])):
    print(f"{f:15s}({len(byf[f])}): " + ", ".join(sorted(byf[f])))
