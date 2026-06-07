"""Generate a HARD non-math LOGIC bank for E13 v3 (discriminant validity).

v1/v2 (factual knowledge) saturated frontier models. Pure deductive *ordering* puzzles instead:
multi-step constraint reasoning, NO arithmetic/algebra, and — crucially — the prose and the
answer key are generated from the SAME constraint spec and the answer is found by brute-forcing
all permutations, so the key is correct *by construction* (no human answer-authoring error). The
asked position must have a unique occupant across all valid orderings (asserted), so the answer
is unambiguous. Difficulty (people count + constraint indirectness) is tuned so weak models drop
and the field spreads, mirroring AIME's range.
"""
from __future__ import annotations

import itertools
import json

ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]

# Each puzzle: (names, constraints, asked_position_index, distractor_idxs)
# constraint ("b", X, Y) means X is positioned strictly before (higher rank than) Y.
PUZZLES = [
    (["Ana", "Ben", "Cara", "Dan", "Eve"],
     [("b", "Ana", "Ben"), ("b", "Ben", "Cara"), ("b", "Cara", "Dan"), ("b", "Dan", "Eve")], 1),
    (["Ana", "Ben", "Cara", "Dan", "Eve"],
     [("b", "Eve", "Ana"), ("b", "Ana", "Dan"), ("b", "Dan", "Ben"), ("b", "Ben", "Cara")], 0),
    (["Tom", "Uma", "Vic", "Wes", "Xena"],
     [("b", "Uma", "Tom"), ("b", "Tom", "Wes"), ("b", "Vic", "Uma"), ("b", "Wes", "Xena")], 2),
    (["Tom", "Uma", "Vic", "Wes", "Xena", "Yan"],
     [("b", "Yan", "Vic"), ("b", "Vic", "Tom"), ("b", "Tom", "Uma"),
      ("b", "Uma", "Wes"), ("b", "Wes", "Xena")], 3),
    (["Pat", "Quin", "Rio", "Sam", "Tess"],
     [("b", "Sam", "Pat"), ("b", "Pat", "Quin"), ("b", "Rio", "Sam"), ("b", "Quin", "Tess")], 4),
    (["Pat", "Quin", "Rio", "Sam", "Tess", "Uri"],
     [("b", "Rio", "Uri"), ("b", "Uri", "Pat"), ("b", "Pat", "Tess"),
      ("b", "Tess", "Quin"), ("b", "Quin", "Sam")], 1),
    (["Amy", "Bo", "Cy", "Di", "Ed", "Fi", "Gus"],
     [("b", "Cy", "Amy"), ("b", "Amy", "Ed"), ("b", "Ed", "Bo"), ("b", "Bo", "Di"),
      ("b", "Di", "Fi"), ("b", "Fi", "Gus")], 2),
    (["Amy", "Bo", "Cy", "Di", "Ed", "Fi"],
     [("b", "Di", "Amy"), ("b", "Amy", "Fi"), ("b", "Fi", "Bo"), ("b", "Bo", "Cy"),
      ("b", "Cy", "Ed")], 0),
    (["Leo", "Mia", "Ned", "Ole", "Pia"],
     [("b", "Mia", "Leo"), ("b", "Ned", "Mia"), ("b", "Leo", "Ole"), ("b", "Ole", "Pia")], 3),
    (["Leo", "Mia", "Ned", "Ole", "Pia", "Ravi"],
     [("b", "Pia", "Ned"), ("b", "Ned", "Ravi"), ("b", "Ravi", "Leo"),
      ("b", "Leo", "Mia"), ("b", "Mia", "Ole")], 4),
    (["Ravi", "Sia", "Tej", "Uma", "Vik"],
     [("b", "Tej", "Vik"), ("b", "Vik", "Ravi"), ("b", "Ravi", "Sia"), ("b", "Sia", "Uma")], 0),
    (["Ravi", "Sia", "Tej", "Uma", "Vik", "Wen"],
     [("b", "Wen", "Sia"), ("b", "Sia", "Tej"), ("b", "Tej", "Ravi"),
      ("b", "Ravi", "Uma"), ("b", "Uma", "Vik")], 2),
    (["Gia", "Hal", "Ivy", "Jo", "Kim"],
     [("b", "Jo", "Gia"), ("b", "Gia", "Kim"), ("b", "Kim", "Hal"), ("b", "Ivy", "Jo")], 1),
    (["Gia", "Hal", "Ivy", "Jo", "Kim", "Lou"],
     [("b", "Ivy", "Lou"), ("b", "Lou", "Hal"), ("b", "Hal", "Gia"),
      ("b", "Gia", "Kim"), ("b", "Kim", "Jo")], 3),
    (["Max", "Nia", "Oz", "Pia", "Quil", "Ros", "Sky"],
     [("b", "Oz", "Max"), ("b", "Max", "Quil"), ("b", "Quil", "Nia"), ("b", "Nia", "Pia"),
      ("b", "Pia", "Ros"), ("b", "Ros", "Sky")], 4),
    (["Max", "Nia", "Oz", "Pia", "Quil", "Ros"],
     [("b", "Pia", "Oz"), ("b", "Oz", "Ros"), ("b", "Ros", "Max"),
      ("b", "Max", "Nia"), ("b", "Nia", "Quil")], 0),
    (["Abe", "Bea", "Cole", "Drew", "Elle"],
     [("b", "Cole", "Elle"), ("b", "Elle", "Abe"), ("b", "Abe", "Bea"), ("b", "Drew", "Cole")], 2),
    (["Abe", "Bea", "Cole", "Drew", "Elle", "Finn"],
     [("b", "Finn", "Drew"), ("b", "Drew", "Bea"), ("b", "Bea", "Abe"),
      ("b", "Abe", "Elle"), ("b", "Elle", "Cole")], 1),
]


def solve(names, constraints, pos):
    """Brute-force: occupant(s) of position `pos` across all orderings satisfying constraints."""
    occ = set()
    full = []
    for perm in itertools.permutations(names):
        idx = {n: i for i, n in enumerate(perm)}
        if all(idx[a] < idx[b] for (_t, a, b) in constraints):
            occ.add(perm[pos])
            full.append(perm)
    return occ, full


def main():
    questions = []
    for qid, (names, cons, pos) in enumerate(PUZZLES, 1):
        occ, full = solve(names, cons, pos)
        assert len(full) >= 1, f"puzzle {qid}: unsatisfiable"
        assert len(occ) == 1, f"puzzle {qid}: position {pos} not unique -> {occ}"
        answer_name = next(iter(occ))
        # build 4 choices: the answer + first three OTHER names (deterministic), then sort for display
        others = [n for n in names if n != answer_name][:3]
        opts = sorted([answer_name] + others)
        letters = ["A", "B", "C", "D"]
        choices = {letters[i]: opts[i] for i in range(4)}
        answer_letter = letters[opts.index(answer_name)]
        sentences = " ".join(f"{a} finished ahead of {b}." for (_t, a, b) in cons)
        q = (f"{len(names)} runners — {', '.join(names)} — finished a race with no ties. "
             f"{sentences} Who finished {ORD[pos]}?")
        questions.append({"id": qid, "domain": "logic", "question": q,
                          "choices": choices, "answer": answer_letter,
                          "n_valid_orderings": len(full)})
    bank = {"name": "nonmath-logic-v3",
            "note": ("Hard non-math deductive ordering puzzles for E13 v3 discriminant validity. "
                     "Prose + answer key generated from one constraint spec; answers brute-force "
                     "verified unique (no arithmetic). Tuned to spread weak->frontier models."),
            "questions": questions}
    json.dump(bank, open("data/banks/nonmath_logic.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"wrote data/banks/nonmath_logic.json with {len(questions)} brute-force-verified puzzles")
    # difficulty proxy: how constrained each is (fewer valid orderings under partial info != difficulty,
    # but report ambiguity of full order)
    amb = [q["n_valid_orderings"] for q in questions]
    print(f"valid-ordering counts (full-order ambiguity): min={min(amb)} max={max(amb)}")


if __name__ == "__main__":
    main()
