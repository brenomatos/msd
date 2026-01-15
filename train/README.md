
# Training scripts and experiments

This folder collects all the scripts used to train and evaluate
classification models.

## Key scripts

| file                         | purpose |
|-----------------------------|---------|
| `main.py`                   | Orchestrates experiments. It parses command‑line arguments, splits the dataset into 5 stratified folds, applies optional undersampling of negative samples, calls the appropriate model implementation (e.g., BERTimbau or PT‑T5), and saves results as a JSONL file. |
| `bertimbau.py`              | Implements a classifier based on the [BERTimbau](https://huggingface.co/neuralmind/bert-base-portuguese-cased) model. It fine‑tunes BERTimbau with a classification head, computes per‑fold metrics (accuracy, precision, recall, F1), and writes each prediction to the output file. |
| `ptt5.py`                   | Implements a classification model using PT‑T5 (Portuguese T5). Similar to `bertimbau.py` but with an encoder–decoder architecture. |
| `llama2.py` and `mistral.py`| Contain optional code for fine‑tuning larger language models (Llama 2 and Mistral) via the [PEFT](https://peft.dev/) library. These scripts use a low‑rank adaptation (LoRA) approach and were used for preliminary experiments; they require significant GPU memory. |
| `metrics.py`                | Reads a results JSONL file and computes aggregate metrics (macro/micro F1, AUROC). |
| `save_as_csv.py`            | Helper script to convert a JSONL results file into a CSV for easier analysis. |
| `temporal-analysis.py`      | Performs *sliding‐window* temporal experiments.  It trains models on an initial set of years and tests on the subsequent year(s) to analyse temporal generalisation.  Both *expand* (train on all previous years) and *walk* (train on a fixed‑length window) strategies are supported. |
| `run.sh`                    | Bash wrapper that runs `main.py` across folds and different class ratios. You can edit this script to schedule multiple experiments. |
| `sliding-window.sh`         | Bash wrapper to automate the temporal experiments; it calls `temporal-analysis.py` with appropriate parameters. |
| `train-models-full.py`      | Additional script to train models on the full dataset without cross‑validation; used for final model fits. |
| `README.md`                 | The original minimal README left by the authors; superseded by this document. |

## Dataset and preparation

The classification scripts expect a CSV file where each row is a
transcript segment with two required fields:

* `text` – the text of the transcript segment;
* `label` – `1` if the segment contains a fact‑checked false claim and `0` otherwise.

An optional `date` column (format `YYYY-MM-DD`) is used by
`temporal-analysis.py` to assign segments to time windows.  For the
BOL4Y dataset we split by the year of the video upload (2019–2023).  The
transcripts themselves are released as CSV files (see
`crawl/escriba-crawler/data/`) and were aligned to claims using the
methodology described in Section 5 of the paper.

To prepare the training data you can use the dataset released via
Zenodo/Hugging Face.

## Running experiments

### Cross‑validation and undersampling

Due to the class imbalance in BOL4Y, we evaluated models using
5‑fold cross‑validation and various undersampling ratios of the
negative class.  In the paper we report results for ratios 1:1, 1:10,
1:25, 1:50, 1:75, 1:100 and the full dataset (i.e., no undersampling). You can reproduce these experiments by editing `run.sh` and then running:

```bash
cd train
bash run.sh
```

The script iterates over the desired ratios and model types, calling
`python main.py` with arguments such as `--dataset-path`, `--fold-index`,
`--ratio`, `--model`, `--batch-size`, `--epochs` and `--results-file`.  See
the top of `run.sh` or run `python main.py --help` for a full list of
arguments.  Example:

```bash
python main.py \
  --dataset-path dataset.csv \
  --fold-index 1 \
  --ratio 10 \
  --model bert \
  --batch-size 32 \
  --epochs 3 \
  --results-file bert-fold1-r10.jsonl
```

After running, use `metrics.py` to summarise the results:

```bash
python metrics.py --results-file bert-fold1-r10.jsonl
```

### Temporal sliding‑window experiments

To analyse how models generalise over time, we perform sliding‑window
experiments on BOL4Y  `temporal-analysis.py` implements these experiments.  For
example, to run PT‑T5 on the BOL4Y dataset with a ratio of 1:75 in the
expand setting:

```bash
python temporal-analysis.py \
  --dataset-path dataset.csv \
  --ratio 75 \
  --model t5 \
  --window expand \
  --results-file t5-expand-r75.jsonl
```

Precomputed results from our paper are available as zipped JSONL files
in `train/results/`: `t5-sliding-window-expand.zip` and
`t5-sliding-window-walk.zip` (for PT‑T5) and
`v2-bert-sliding-ratio75-window-expand.jsonl` and
`v2-bert-sliding-ratio75-window-walk.jsonl` (for BERTimbau at a ratio of
75).  

## Hardware requirements

Fine‑tuning BERTimbau and PT‑T5 on the full dataset requires a
GPU with at least 16 GB memory.  Training Llama 2 or Mistral via
LoRA uses even more memory.
Temporal experiments can be memory‑intensive because they involve
multiple training iterations.  Adjust the batch size and number of
epochs according to your hardware.
