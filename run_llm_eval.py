"""
Compares the regex-based classifier against an LLM-based one, on the same
held-out scenarios, so the two numbers are directly comparable.

Two providers supported:
  --provider gemini     (default) needs GEMINI_API_KEY, genuinely free,
                         no card required -- see gemini_classify.py
  --provider anthropic  needs ANTHROPIC_API_KEY, paid past a small trial
                         credit -- see llm_classify.py

Runs on a smaller sample than the main 200 by default -- each scenario means
one or more real API calls, so this takes a minute or two. Bump --n once
you're happy it works.

Usage:
  python run_llm_eval.py                        # gemini, 60 scenarios
  python run_llm_eval.py --n 200                 # gemini, full set
  python run_llm_eval.py --provider anthropic    # anthropic instead
"""

import argparse
import os
import sys

from generator import generate
from score import evaluate, naive_baseline
from classify import classify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60,
                        help="number of held-out scenarios to score")
    parser.add_argument("--provider", choices=["gemini", "anthropic"],
                        default="gemini",
                        help="which LLM to use for injection detection "
                             "(gemini is free, no card needed)")
    args = parser.parse_args()

    if args.provider == "gemini":
        key_name = "GEMINI_API_KEY"
        setup_url = "https://aistudio.google.com"
    else:
        key_name = "ANTHROPIC_API_KEY"
        setup_url = "https://console.anthropic.com"

    if not os.environ.get(key_name):
        print(f"{key_name} is not set.")
        print(f"Get a key at {setup_url}, then:")
        print(f"  Windows cmd:   set {key_name}=...")
        print(f"  PowerShell:    $env:{key_name}=\"...\"")
        sys.exit(1)

    if args.provider == "gemini":
        from gemini_classify import classify_llm
    else:
        from llm_classify import classify_llm

    scenarios = generate(args.n, held_out=True)

    print(f"Scoring {args.n} held-out scenarios, three ways "
          f"(LLM provider: {args.provider}).\n")
    print("This calls the API once per merchant-content input in each "
          "scenario -- give it a minute.\n")

    evaluate(scenarios, naive_baseline, label="naive baseline (held-out)")
    evaluate(scenarios, classify, label="rule engine, regex injection (held-out)")
    evaluate(scenarios, classify_llm,
             label=f"rule engine, {args.provider} injection (held-out)")


if __name__ == "__main__":
    main()
