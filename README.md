
# Misinformation Span Detection

This repository contains the
scripts used to collect the data, preprocess transcripts, and train
classification models for detecting which segments of a video contain
misinformative claims.  The repository is split into two major
modules:

* **`crawl/`** – Scripts for scraping fact‑checked claims from the
  **Aos Fatos** portal, downloading videos, and fetching transcripts.
* **`train/`** – Python scripts for preparing the dataset,
  fine‑tuning language models, evaluating performance, and conducting
  temporal experiments.

## Repository structure

| path          | description |
|--------------|-------------|
| `crawl/`     | Code to download the raw data.  See `/crawl/README.md` (in this repository) for details on crawling, parsing HTML pages, downloading videos, and retrieving Escriba transcripts. |
| `train/`     | Code to prepare the dataset and run experiments.  See `/train/README.md` for instructions on cross‑validation, undersampling, temporal sliding‑window experiments, and model training (BERTimbau, PT‑T5, Llama 2, Mistral). |
| `LICENSE`    | Consult this file for terms of use. |

Important subfolders inside `train/` and `crawl/` each contain their own
README files explaining the specifics of those modules.  For example,
`crawl/aos-fatos/pages` stores the cached HTML pages used to parse
claims, while `train/results` holds the JSONL results of our
experiments.  You can find the corresponding README files (in
plain‑text format) in the repository alongside this document.  

## Getting started

To reproduce the dataset or run your own experiments, follow these
steps (details are elaborated in the subfolder READMEs):

1. **Clone the repository** and install Python 3.9+ along with the
   required packages (`requests`, `beautifulsoup4`, `yt_dlp`,
   `playwright`, `pandas`, `transformers`, `torch`, and
   `peft`).  The `train` scripts require GPU acceleration for
   efficient training.
2. **Crawl the data** using the scripts in `crawl/aos-fatos/`.  This
   will download claim pages, parse them into structured data, and
   download the corresponding videos.  If you already have access to
   the preprocessed CSVs released on Zenodo, you may skip this step.
3. **Download Escriba transcripts** for claims without video
   sources via `crawl/escriba-crawler/escriba.py`.
4. **Prepare the training dataset** using the CSV files shared on the Zenodo
   release.  Each row should contain the transcript text, its label
   (1 for false claim, 0 otherwise), and the date of the video.
5. **Run experiments** using the scripts in `train/`: you can perform
   5‑fold cross‑validation with various undersampling ratios, fine‑tune
   models such as BERTimbau and PT‑T5, and evaluate temporal
   generalisation using sliding‑window experiments.

## Citation and license

tba
