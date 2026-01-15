# llm-experiments
LLMs for misinformation detection

# Running
Define your experiments on the run.sh file. You'll need to set the following arguments:

- dataset-path: Path to a csv file containing all instances
- fold-index: Index of fold for cross validation. Ranges from 1 to 5
- run-complete-dataset: A flag to signal that we wont use the undersampling ratio option (true/false)
- ratio: Ratio of positive-to-negative examples (use 1, 10, or 100)
- model: Which model to run. Options: bert, t5, llama
- batch-size: Batch size
- epochs: Number of epochs
- results-file: JSONL file to store results

# Scripts
- bertimbau.py and ptt5.py define functions to run respective models
- main.py handles parsing input and calling models
- run.sh is a bash script to run fold-by-fold
- metrics.py calculates average values given a results file
- save_as_csv.py concatenates results into a single file 
- temporal-analysis.py performs the temporal experiments described in the paper
- sliding-window.sh is a script that facilitated running temporal-analysis.py
