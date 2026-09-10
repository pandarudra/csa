#!/usr/bin/env python3
"""Run the full SpotifyCares support agent on one message and print its output.

Usage:
    python scripts/run_pipeline_demo.py "I can't play any songs on Premium"
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import SupportAgent  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} \"<customer message>\"", file=sys.stderr)
        raise SystemExit(1)

    message = sys.argv[1]
    agent = SupportAgent.load()
    result = agent.handle(message)

    print(f"\nCustomer message:\n  {message}\n")
    print(f"Intent:      {result.intent}")
    print(f"Confidence:  {result.intent_confidence:.2f}\n")

    print("Retrieved historical cases:")
    if result.retrieved_examples:
        for example in result.retrieved_examples:
            print(f"  [{example.similarity:.2f}] {example.customer_problem!r}")
            print(f"         -> {example.brand_reply!r}")
    else:
        print("  (none)")

    print(f"\nDraft reply:\n  {result.reply}\n")
    print(f"Decision: {result.action.value}")
    print(f"Reason:   {result.escalation_reason}")


if __name__ == "__main__":
    main()
