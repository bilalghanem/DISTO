"""DISTO scorer: a learned metric for the quality of MCQ distractors."""

import re
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL = "bilalghanem/DISTO"
MAX_DISTRACTORS = 3
EMPTY_TOKEN = "[EMPT]"


def build_input(
    article: str,
    question: str,
    answer: str,
    distractors: Sequence[Optional[str]],
) -> str:
    """Format one MCQ the way the model was trained.

    The layout is ``[QUES] q [ANS] a [DIS1] d1 [DIS2] d2 [DIS3] d3 [ART] article``.
    Missing distractors are filled with ``[EMPT]``. The article goes last so that
    truncation to the maximum length only ever cuts the article.
    """
    distractors = list(distractors)
    if len(distractors) > MAX_DISTRACTORS:
        raise ValueError(
            f"DISTO scores at most {MAX_DISTRACTORS} distractors per question, "
            f"got {len(distractors)}."
        )
    distractors += [None] * (MAX_DISTRACTORS - len(distractors))
    slots = [d if d and d.strip() else EMPTY_TOKEN for d in distractors]

    parts = [f"[QUES] {question}", f"[ANS] {answer}"]
    parts += [f"[DIS{i}] {d}" for i, d in enumerate(slots, start=1)]
    parts.append(f"[ART] {article}")
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


class DistoScorer:
    """Scores a set of distractors for a multiple-choice reading comprehension question.

    The score lies in [0, 1]. Higher means the distractors are more plausible
    given the article, question and correct answer; a distractor that copies the
    answer or has nothing to do with the context is pushed towards 0.

    Example:
        >>> scorer = DistoScorer()
        >>> scorer.score(
        ...     article="Paris is the capital of France. ...",
        ...     question="What is the capital of France?",
        ...     answer="Paris",
        ...     distractors=["Lyon", "Marseille", "Toulouse"],
        ... )
    """

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
            .to(self.device)
            .eval()
        )

    @torch.no_grad()
    def score_batch(
        self, instances: Iterable[Dict], batch_size: int = 16
    ) -> List[float]:
        """Score many MCQs. Each instance is a dict with the keys
        ``article``, ``question``, ``answer`` and ``distractors`` (a list of up to 3 strings)."""
        texts = [
            build_input(
                inst["article"], inst["question"], inst["answer"], inst["distractors"]
            )
            for inst in instances
        ]
        scores: List[float] = []
        for start in range(0, len(texts), batch_size):
            batch = self.tokenizer(
                texts[start : start + batch_size],
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**batch).logits.squeeze(-1)
            scores.extend(torch.sigmoid(logits).reshape(-1).tolist())
        return scores

    def score_each(
        self,
        article: str,
        question: str,
        answer: str,
        distractors: Sequence[str],
    ) -> List[float]:
        """Score every distractor on its own (the protocol used in the paper).

        Each distractor is placed alone in the first slot, with the other two slots
        empty, and the model is run once per distractor. The paper averages these
        scores to rate a set of generated distractors: ``sum(s) / len(s)``.
        """
        return self.score_batch(
            [
                {
                    "article": article,
                    "question": question,
                    "answer": answer,
                    "distractors": [d],
                }
                for d in distractors
            ]
        )

    def score(
        self,
        article: str,
        question: str,
        answer: str,
        distractors: Sequence[Optional[str]],
    ) -> float:
        """Score the distractors of a single MCQ."""
        return self.score_batch(
            [
                {
                    "article": article,
                    "question": question,
                    "answer": answer,
                    "distractors": distractors,
                }
            ]
        )[0]
