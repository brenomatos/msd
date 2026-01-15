python3 temporal-analysis.py --sliding-type "walk-forward" --model bert --dataset-path slim-dataset.csv --sliding-step 1 --test-length 1 --train-length 6 --ratio 100 --batch-size 32 --epochs 3 --results-file bert-sliding-ratio100-window-walk.jsonl

python3 temporal-analysis.py --sliding-type "expand-test" --model bert --dataset-path slim-dataset.csv --sliding-step 1 --test-length 1 --train-length 6 --ratio 100 --batch-size 32 --epochs 3 --results-file bert-sliding-ratio100-window-expand.jsonl

python3 temporal-analysis.py --sliding-type "walk-forward" --model t5 --dataset-path slim-dataset.csv --sliding-step 1 --test-length 1 --train-length 6 --ratio 10 --batch-size 32 --epochs 3 --results-file t5-sliding-ratio10-window-walk.jsonl

python3 temporal-analysis.py --sliding-type "expand-test" --model t5 --dataset-path slim-dataset.csv --sliding-step 1 --test-length 1 --train-length 6 --ratio 10 --batch-size 32 --epochs 3 --results-file t5-sliding-ratio10-window-expand.jsonl
