"""Command line interface: ``python -m disto input.jsonl -o scores.jsonl``."""

import argparse
import json
import sys

from .scorer import DEFAULT_MODEL, DistoScorer


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="disto",
        description="Score MCQ distractors with DISTO. The input is a JSONL file where each "
        "line has the keys: article, question, answer, distractors (list of up to 3).",
    )
    parser.add_argument("input", help="input JSONL file ('-' for stdin)")
    parser.add_argument("-o", "--output", default="-", help="output JSONL file (default: stdout)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id or local path")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None, help="e.g. cpu, cuda, mps (default: auto)")
    args = parser.parse_args()

    src = sys.stdin if args.input == "-" else open(args.input, encoding="utf-8")
    with src:
        instances = [json.loads(line) for line in src if line.strip()]

    scorer = DistoScorer(args.model, device=args.device)
    scores = scorer.score_batch(instances, batch_size=args.batch_size)

    out = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8")
    with out:
        for inst, score in zip(instances, scores):
            out.write(json.dumps({**inst, "disto_score": round(score, 4)}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
