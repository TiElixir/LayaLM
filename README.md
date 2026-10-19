# Laya recursive text generator

Requires Python 3.11 and Conda. In PowerShell, from this folder:

```powershell
conda create -n laya python=3.11 -y
conda activate laya
python -m pip install -r requirements.txt
python main.py
```

Enter a prompt and the number of words to append. The vocabulary is read from
the local `.nltk_data/corpora/words/en1k` file. Put or replace your own
newline-separated word list at that path; the program does not download or
modify vocabulary data. Laya's checkpoint is downloaded on first inference.
You may then choose to save the result as a `.txt` file.

The en1k vocabulary is larger than Laya's choice-option budget. The script
compares batches of 100 and carries each winner forward to a tournament.
The official Laya README gives a practical limit of about 126 short options for
the English model, with an exact limit that depends on the option text and
instructions. With the current en1k file, each generated word requires about
100 Laya decisions plus comparison rounds. Batch winners make this an approximate tournament
rather than a single global ranking.

Laya accepts a `choice` question through `Router.predict(state, questions)`.
Its English checkpoint has a 512-token context window and uses part of that
budget for choice options, so very long prompts may be truncated by the model.

After each choice, the script adjusts Laya's per-word probabilities by dividing
the probability of a word already used in the text by `10` for each prior
occurrence (capped at twenty applications), then renormalizes and picks the
highest adjusted score. This reduces repeated words; it does not ban reused
letters, since English words necessarily share letters.

Each decision also receives the fixed `SYSTEM_PROMPT` instruction appended to
the current text sent to Laya. The instruction is not included in the output.
