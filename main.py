"""A simple recursive text generator driven by Laya choice decisions."""

from __future__ import annotations

import sys
import re
from pathlib import Path

from laya import Router


BASE_DIR = Path(__file__).resolve().parent
VOCABULARY_FILE = BASE_DIR / ".nltk_data" / "corpora" / "words" / "en1k"
# Laya's README documents a practical limit of about 126 short options on its
# English checkpoint (the exact limit depends on labels and instructions).
BATCH_SIZE = 100
# Strongly discount a word already present in the prompt or generated text.
REPETITION_PENALTY = 10.0
SYSTEM_PROMPT = (
    "Continue the user's text naturally. Choose the single most suitable English word to append next. Do not repeat a word unless it fits naturally. "
)


def load_vocabulary() -> list[str]:
    """Read the manually supplied en1k word list in its existing order."""
    if not VOCABULARY_FILE.is_file():
        raise FileNotFoundError(f"Local vocabulary file not found: {VOCABULARY_FILE}")
    vocabulary = [
        line.strip().lower()
        for line in VOCABULARY_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    vocabulary = list(dict.fromkeys(vocabulary))
    if not vocabulary:
        raise ValueError(f"Local vocabulary file is empty: {VOCABULARY_FILE}")
    return vocabulary


def apply_repetition_penalty(
    probabilities: dict[str, float], text: str, penalty: float = REPETITION_PENALTY
) -> dict[str, float]:
    """Reduce probabilities for words already used; keep scores comparable."""
    if penalty < 1.0:
        raise ValueError("Repetition penalty must be at least 1.0")
    counts: dict[str, int] = {}
    for word in re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower()):
        counts[word] = counts.get(word, 0) + 1

    adjusted = {
        word: probability / (penalty ** min(counts.get(word.lower(), 0), 20))
        for word, probability in probabilities.items()
    }
    total = sum(adjusted.values())
    if total <= 0:
        return probabilities
    return {word: score / total for word, score in adjusted.items()}


def choose_from_batch(router: Router, state: str, candidates: list[str]) -> str:
    """Ask Laya to choose a candidate, then penalize previously used words."""
    if len(candidates) == 1:
        return candidates[0]
    choices = {
        word: f"The next word should be '{word}'."
        for word in candidates
    }
    result = router.predict(
        f"{state}\n\n{SYSTEM_PROMPT}",
        {"next_word": {
            "type": "choice",
            "instructions": "Which candidate is the most suitable next word?",
            "criteria": choices,
        }},
        model="english",
    )
    answer = result["answers"]["next_word"]
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict) or not probabilities:
        raise ValueError("Laya did not return choice probabilities")
    if any(word not in choices for word in probabilities):
        raise ValueError("Laya returned probabilities for an invalid candidate")
    adjusted = apply_repetition_penalty(probabilities, state)
    prediction = max(adjusted, key=adjusted.get)
    used_words = set(re.findall(r"[a-z]+(?:'[a-z]+)?", state.lower()))
    unused = [word for word in adjusted if word.lower() not in used_words]
    if prediction.lower() in used_words and unused:
        prediction = max(unused, key=adjusted.get)
    return prediction


def choose_next_word(router: Router, state: str, vocabulary: list[str]) -> str:
    """Run a bounded-choice tournament so every vocabulary word is considered."""
    winners = [
        choose_from_batch(router, state, vocabulary[start : start + BATCH_SIZE])
        for start in range(0, len(vocabulary), BATCH_SIZE)
    ]
    while len(winners) > 1:
        winners = [
            choose_from_batch(router, state, winners[start : start + BATCH_SIZE])
            for start in range(0, len(winners), BATCH_SIZE)
        ]
    return winners[0]


def main() -> int:
    try:
        prompt = input("Enter a text prompt: ").strip()
        if not prompt:
            print("Prompt cannot be empty.", file=sys.stderr)
            return 1
        try:
            word_limit = int(input("Maximum number of generated words: ").strip())
        except ValueError:
            print("Word limit must be a whole number.", file=sys.stderr)
            return 1
        if word_limit < 0:
            print("Word limit cannot be negative.", file=sys.stderr)
            return 1

        vocabulary = load_vocabulary()
        print(f"Loaded {len(vocabulary):,} English words. Initializing Laya...")
        router = Router()
        generated = prompt
        for index in range(word_limit):
            print(f"Selecting word {index + 1}/{word_limit}...", flush=True)
            word = choose_next_word(router, generated, vocabulary)
            generated += " " + word

        print("\nGenerated text:\n" + generated)
        save = input("Save to a .txt file? [y/N]: ").strip().lower()
        if save in {"y", "yes"}:
            filename = input("Filename [generated.txt]: ").strip() or "generated.txt"
            path = Path(filename)
            if path.suffix.lower() != ".txt":
                path = path.with_suffix(".txt")
            path.write_text(generated + "\n", encoding="utf-8")
            print(f"Saved to {path.resolve()}")
        return 0
    except KeyboardInterrupt:
        print("\nGeneration cancelled.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
