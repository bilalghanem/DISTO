from disto import DistoScorer

scorer = DistoScorer()  # downloads the model from the Hugging Face Hub on first use

article = (
    "I started freaking out, but I couldn't really get help because I was in the middle of "
    "nowhere, so I just kept taking pictures hoping that it would work. Eventually the battery "
    "ran out and I didn't turn it on again until later that night. When I did, it displayed the "
    "default background instead of the picture of my dog that I had on it earlier in the day. "
    "It immediately went to a message about formatting it and the information being lost, so "
    "naturally, I said no."
)
question = "What may happen after their camera malfunctioned?"
answer = "They will try to retrieve their lost photos"

candidates = {
    "plausible": ["They will find someone to fix their camera", "They will call someone to fix their camera"],
    "copies the answer": [answer, "They will find someone to fix their camera"],
    "unrelated": ["The moon is made of cheese", "Paris is the capital of France"],
}

for name, distractors in candidates.items():
    print(f"{name:>18}: {scorer.score(article, question, answer, distractors):.3f}")
