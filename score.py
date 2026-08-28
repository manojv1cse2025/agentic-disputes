"""
Scoring harness for dispute classifiers.

Takes any function of the shape  classify(public_scenario_dict) -> Verdict
and reports:

  1. overall accuracy
  2. a confusion matrix
  3. FALSE BLAME: how often an innocent merchant was blamed. This is the
     expensive error and it gets its own number.
  4. MISSED INJECTION: a real manipulation that went undetected
  5. a per-generator breakdown, so failures point at a specific case type

Run this file directly to score the naive baseline. The baseline is here to
prove the harness discriminates: it looks respectable on accuracy while
failing badly on false blame.
"""

from collections import defaultdict
from typing import Callable, Dict, List

from generator import Scenario, Verdict, generate


ClassifierFn = Callable[[dict], Verdict]

ALL = [Verdict.WITHIN_MANDATE, Verdict.MANDATE_BREACH,
       Verdict.INJECTED, Verdict.AMBIGUOUS]


def evaluate(scenarios: List[Scenario], classify: ClassifierFn,
             label: str = "classifier") -> Dict:
    confusion: Dict = defaultdict(lambda: defaultdict(int))
    by_generator: Dict = defaultdict(lambda: {"n": 0, "correct": 0})
    false_blame = []
    missed_injection = []

    for s in scenarios:
        pred = classify(s.to_public_dict())
        truth = s.truth
        confusion[truth][pred] += 1

        g = by_generator[s.generator]
        g["n"] += 1
        if pred == truth:
            g["correct"] += 1

        # Blaming a third party when nobody manipulated anything.
        if pred == Verdict.INJECTED and truth != Verdict.INJECTED:
            false_blame.append(s)

        # A real manipulation the classifier did not catch.
        if truth == Verdict.INJECTED and pred != Verdict.INJECTED:
            missed_injection.append(s)

    correct = sum(confusion[v][v] for v in ALL)
    n = len(scenarios)

    print(f"\n{'=' * 62}")
    print(f"  {label}   n = {n}")
    print(f"{'=' * 62}")
    print(f"\n  accuracy            {correct}/{n}  ({correct / n:.1%})")
    print(f"  false blame         {len(false_blame)}  "
          f"(innocent merchant blamed for manipulation)")
    print(f"  missed injection    {len(missed_injection)}  "
          f"(real manipulation not caught)")

    print(f"\n  confusion matrix   rows = truth, cols = predicted")
    head = "".join(f"{v.value[:9]:>11}" for v in ALL)
    print(f"    {'':<17}{head}")
    for t in ALL:
        row = "".join(f"{confusion[t][p]:>11}" for p in ALL)
        print(f"    {t.value:<17}{row}")

    print(f"\n  by case type")
    for gen, st in sorted(by_generator.items(),
                          key=lambda kv: kv[1]["correct"] / max(kv[1]["n"], 1)):
        rate = st["correct"] / st["n"]
        flag = "  <-- weakest" if rate < 0.5 else ""
        print(f"    {gen:<22} {st['correct']:>3}/{st['n']:<4} {rate:>6.0%}{flag}")
    print()

    return {
        "n": n,
        "accuracy": correct / n,
        "false_blame": len(false_blame),
        "missed_injection": len(missed_injection),
        "by_generator": dict(by_generator),
    }


# --------------------------------------------------------------------------
# baseline: the classifier a person writes in twenty minutes
# --------------------------------------------------------------------------

INJECTION_KEYWORDS = ["system", "override", "ignore", "admin", "instruction"]


def naive_baseline(sc: dict) -> Verdict:
    """
    Flags injection on keywords, otherwise checks the per-transaction cap only.
    Deliberately shallow. It should fail cumulative_breach (only looks at this
    transaction), subtle_injection (no keywords), and injection_ignored (flags
    on presence of text rather than on whether the agent acted on it).
    """
    blob = " ".join(i["text"].lower() for i in sc["agent_inputs"])
    if any(k in blob for k in INJECTION_KEYWORDS):
        return Verdict.INJECTED

    cap = sc["mandate"]["per_txn_cap"]
    if cap is not None and sc["transaction"]["amount"] > cap:
        return Verdict.MANDATE_BREACH

    if sc["transaction"]["category"] not in sc["mandate"]["allowed_categories"]:
        return Verdict.MANDATE_BREACH

    return Verdict.WITHIN_MANDATE


if __name__ == "__main__":
    scenarios = generate(200)
    evaluate(scenarios, naive_baseline, label="naive baseline")
