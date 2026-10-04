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
    parser.add_argument(
        "--each",
        action="store_true",
        help="score every distractor on its own (the paper's protocol); adds disto_scores and "
        "disto_score (their mean) instead of one score for the whole set",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None, help="e.g. cpu, cuda, mps (default: auto)")
    args = parser.parse_args()

    src = sys.stdin if args.input == "-" else open(args.input, encoding="utf-8")
    with src:
        instances = [json.loads(line) for line in src if line.strip()]

    scorer = DistoScorer(args.model, device=args.device)
    out = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8")
    with out:
        if args.each:
            # one single-distractor instance per distractor, scored together in batches
            flat = [
                {**inst, "distractors": [d]} for inst in instances for d in inst["distractors"]
            ]
            flat_scores = iter(scorer.score_batch(flat, batch_size=args.batch_size))
            for inst in instances:
                scores = [round(next(flat_scores), 4) for _ in inst["distractors"]]
                mean = round(sum(scores) / len(scores), 4) if scores else None
                record = {**inst, "disto_scores": scores, "disto_score": mean}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            scores = scorer.score_batch(instances, batch_size=args.batch_size)
            for inst, score in zip(instances, scores):
                out.write(json.dumps({**inst, "disto_score": round(score, 4)}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
