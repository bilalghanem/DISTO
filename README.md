# DISTO: a learned metric for multiple-choice distractors

DISTO scores how good the **distractors** of a multiple-choice reading comprehension question are, given the
article, the question and the correct answer. It does not compare against reference distractors, so it
does not penalise a distractor for being different from the gold one (which is what BLEU/ROUGE-style
metrics do).

This repository contains the **scorer**. The trained model is hosted on the Hugging Face Hub:
[`bilalghanem/DISTO`](https://huggingface.co/bilalghanem/DISTO).

> **Paper:** Bilal Ghanem and Alona Fyshe. *DISTO: Textual Distractors for Multiple Choice Reading
> Comprehension Questions Using Negative Sampling.* EDM 2024.
> [`docs/DISTO_EDM2024.pdf`](docs/DISTO_EDM2024.pdf) · [doi:10.5281/zenodo.12729766](https://doi.org/10.5281/zenodo.12729766)

## Installation

```bash
git clone https://github.com/bilalghanem/DISTO.git
cd DISTO
pip install .
```

Requires Python 3.8+, PyTorch and 🤗 Transformers. A GPU is optional.

## Quick start

```python
from disto import DistoScorer

scorer = DistoScorer()  # downloads bilalghanem/DISTO from the Hub on first use

score = scorer.score(
    article="The Nile is the longest river in Africa. It flows north through Egypt into the Mediterranean Sea.",
    question="Where does the Nile flow into?",
    answer="The Mediterranean Sea",
    distractors=["The Red Sea", "The Atlantic Ocean", "Lake Victoria"],
)
print(score)  # ≈ 0.99
```

A copy of the answer or an unrelated distractor gets a low score (see [`examples/score_example.py`](examples/score_example.py)).

### Scoring one distractor at a time (the paper's protocol)

The paper rates each distractor on its own and averages the scores for a set of generated distractors.
`score_each` does exactly that: every distractor goes alone in the first slot, with the other two slots empty.

```python
scores = scorer.score_each(article, question, answer, ["The Red Sea", "The Atlantic Ocean", "Lake Victoria"])
paper_score = sum(scores) / len(scores)
```

`score` instead rates the whole set of up to three distractors in one pass, which is what the model was trained on.

Score many questions at once:

```python
scores = scorer.score_batch(
    [{"article": ..., "question": ..., "answer": ..., "distractors": [...]}, ...],
    batch_size=16,
)
```

Use a different checkpoint or a local folder with `DistoScorer("path/or/hub-id")`.

### Command line

```bash
python -m disto examples/input.jsonl -o scores.jsonl
```

Each input line is a JSON object with `article`, `question`, `answer` and `distractors`. The output repeats the
line and adds a `disto_score` field.

## How it works

DISTO is a `distilroberta-base` encoder with a one-output regression head and a sigmoid. The input is

```
[QUES] question [ANS] answer [DIS1] d1 [DIS2] d2 [DIS3] d3 [ART] article
```

truncated to 512 tokens (only the article is ever cut). A question with fewer than three distractors is
padded with `[EMPT]`; more than three is rejected. The paper writes this input with a single `[DIS]` slot;
the released model has three slots, and the single-distractor case is a distractor in slot 1 with the other
two empty. Training includes many such one-distractor examples.

The model is trained with negative sampling: good distractors from seven reading comprehension datasets
(CosmosQA, DREAM, MCScript, MCTest, QuAIL, RACE, SciQ) get a target of 1, and distractors replaced with a
copy of the answer, a random distractor, the farthest point in a k-means cluster, or a BERT
`[MASK]`-filled rewrite pull the target down. The output is a score in `[0, 1]`.

### Results

On the held-out test split of this release (66,322 instances):

| Metric | Value |
|---|---|
| MAE | 0.0286 |
| Pearson correlation | 0.966 |

The paper reports 0.038 MAE / 0.941 Pearson for the DISTO setup in its Table 3, and a Pearson correlation of
0.81 between DISTO and Amazon Mechanical Turk ratings (Table 4).

## Citation

```bibtex
@inproceedings{ghanem2024disto,
  title     = {{DISTO}: Textual Distractors for Multiple Choice Reading Comprehension Questions Using Negative Sampling},
  author    = {Ghanem, Bilal and Fyshe, Alona},
  booktitle = {Proceedings of the 17th International Conference on Educational Data Mining},
  pages     = {6--17},
  year      = {2024},
  address   = {Atlanta, Georgia, USA},
  publisher = {International Educational Data Mining Society},
  doi       = {10.5281/zenodo.12729766}
}
```

## License

The code is released under the [Apache License 2.0](LICENSE). The paper in `docs/` is distributed under
CC BY-NC-ND 4.0. The model was trained on public reading comprehension datasets, each with its own terms
of use; check them before using the model commercially.
