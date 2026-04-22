# Data Layout

This project uses public-domain Sherlock Holmes texts from Project Gutenberg.
The raw source texts are downloaded by code so the Colab workflow does not rely
on manual local setup.

## Source Corpus

- The Adventures of Sherlock Holmes
- The Memoirs of Sherlock Holmes
- The Hound of the Baskervilles
- The Return of Sherlock Holmes

The source metadata is defined in `src/novel_continuation/data_sources.py`.

## Directory Layout

- `data/raw/`: downloaded raw Project Gutenberg `.txt` files
- `data/processed/`: cleaned paragraph-level datasets built by later scripts

## Downloading the Raw Data

From the repository root:

```bash
python scripts/download_gutenberg.py --output-dir data/raw
```

In Colab, the same script can be run after cloning the repository and
installing dependencies. The download locations are controlled through the
`--output-dir` argument so notebooks can write to Drive-backed paths if needed.
The script is safe to run repeatedly: existing files are skipped by default,
and `--force` can be used to re-download them.
