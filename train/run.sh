# ############## BERT MODELS

for i in {1..5}
do
 # ratio 1
   python3 main.py --model bert --dataset-path slim-dataset-v2.csv.gz --fold-index $i --run-complete-dataset false --ratio 1 --batch-size 32 --epochs 3 --results-file results/v2-slim-bert-results-ratio1.jsonl
   rm -rf my_awesome_model
done

for i in {1..5}
do
 # ratio 10
   python3 main.py --model bert --dataset-path slim-dataset-v2.csv.gz --fold-index $i --run-complete-dataset false --ratio 10 --batch-size 32 --epochs 3 --results-file results/v2-slim-bert-results-ratio10.jsonl
   rm -rf my_awesome_model
done

for i in {1..5}
do
 # ratio 25
   python3 main.py --model bert --dataset-path slim-dataset-v2.csv.gz --fold-index $i --run-complete-dataset false --ratio 25 --batch-size 32 --epochs 3 --results-file results/v2-slim-bert-results-ratio25.jsonl
   rm -rf my_awesome_model
done

for i in {1..5}
do
 # ratio 50
   python3 main.py --model bert --dataset-path slim-dataset-v2.csv.gz --fold-index $i --run-complete-dataset false --ratio 50 --batch-size 32 --epochs 3 --results-file results/v2-slim-bert-results-ratio50.jsonl
   rm -rf my_awesome_model
done

#for i in {1..5}
#do
# # ratio 75
#   python3 main.py --model bert --dataset-path slim-dataset-v2.csv.gz --fold-index $i --run-complete-dataset false --ratio 75 --batch-size 32 --epochs 3 --results-file results/v2-slim-bert-results-ratio75.jsonl
#   rm -rf my_awesome_model
#done


for i in {1..5}
do
 # ratio 100
  python3 main.py --model bert --dataset-path slim-dataset-v2.csv.gz --fold-index $i --run-complete-dataset false --ratio 100 --batch-size 32 --epochs 3 --results-file results/v2-slim-bert-results-ratio100.jsonl
  rm -rf my_awesome_model
done

for i in {1..5}
do
 # full dataset
   python3 main.py --model bert --dataset-path slim-dataset-v2.csv.gz --fold-index $i --run-complete-dataset true --ratio 0 --batch-size 32 --epochs 3 --results-file results/v2-slim-bert-results-full.jsonl
   rm -rf my_awesome_model
done

# for i in {1..5}
# do
#  # ratio 125
#   python3 main.py --model bert --dataset-path slim-dataset-v2.csv.gz --fold-index $i --run-complete-dataset false --ratio 125 --batch-size 32 --epochs 3 --results-file results/slim-bert-results-ratio125.jsonl
#   rm -rf my_awesome_model
# done

# for i in {1..5}
# do
#  # ratio 500
#   python3 main.py --model bert --dataset-path slim-dataset-v2.csv.gz --fold-index $i --run-complete-dataset false --ratio 500 --batch-size 32 --epochs 3 --results-file results/slim-bert-results-ratio500.jsonl
#   rm -rf my_awesome_model
# done

# for i in {1..5}
# do
#  # ratio 400
#   python3 main.py --model bert --dataset-path slim-dataset-v2.csv.gz --fold-index $i --run-complete-dataset false --ratio 400 --batch-size 32 --epochs 3 --results-file results/slim-bert-results-ratio400.jsonl
#   rm -rf my_awesome_model
# done

# for i in {1..5}
# do
#  # ratio 300
#   python3 main.py --model bert --dataset-path slim-dataset-v2.csv.gz --fold-index $i --run-complete-dataset false --ratio 300 --batch-size 32 --epochs 3 --results-file results/slim-bert-results-ratio300.jsonl
#   rm -rf my_awesome_model
# done





# # ############# T5 MODELS

for i in {1..5}
do
#    # ratio 1
   python3 main.py --model t5 --dataset-path slim-dataset-v2.csv.gz --fold-index $i --run-complete-dataset false --ratio 1 --batch-size 32 --epochs 3 --results-file results/slim-t5-results-ratio1.jsonl
   rm -rf my_awesome_model
done

for i in {1..5}
do
#    # ratio 10
   python3 main.py --model t5 --dataset-path slim-dataset-v2.csv.gz --fold-index $i --run-complete-dataset false --ratio 10 --batch-size 32 --epochs 3 --results-file results/slim-t5-results-ratio10.jsonl
   rm -rf my_awesome_model
done

for i in {1..5}
do
     # ratio 25
   python3 main.py --model t5 --dataset-path slim-dataset-v2.csv.gz --fold-index $i --run-complete-dataset false --ratio 25 --batch-size 32 --epochs 3 --results-file results/v2-slim-t5-results-ratio25.jsonl
   rm -rf my_awesome_model
done

for i in {1..5}
do
     # ratio 50
   python3 main.py --model t5 --dataset-path slim-dataset-v2.csv.gz --fold-index $i --run-complete-dataset false --ratio 50 --batch-size 32 --epochs 3 --results-file results/v2-slim-t5-results-ratio50.jsonl
   rm -rf my_awesome_model
done

for i in {1..5}
do
     # ratio 75
   python3 main.py --model t5 --dataset-path slim-dataset-v2.csv.gz --fold-index $i --run-complete-dataset false --ratio 75 --batch-size 32 --epochs 3 --results-file results/v2-slim-t5-results-ratio75.jsonl
   rm -rf my_awesome_model
done



for i in {1..5}
do
#    # ratio 100
   python3 main.py --model t5 --dataset-path slim-dataset-v2.csv.gz --fold-index $i --run-complete-dataset false --ratio 100 --batch-size 32 --epochs 3 --results-file results/v2-slim-t5-results-ratio100.jsonl
   rm -rf my_awesome_model
done


for i in {1..5}
do
#    # FULL
   python3 main.py --model t5 --dataset-path slim-dataset-v2.csv.gz --fold-index $i --run-complete-dataset true --ratio 0 --batch-size 32 --epochs 3 --results-file results/v2-slim-t5-results-full.jsonl
   rm -rf my_awesome_model
done

# for i in {1..5}
# do
# #    # ratio 125
#    python3 main.py --model t5 --dataset-path slim-dataset-v2.csv.gz --fold-index $i --run-complete-dataset false --ratio 125 --batch-size 32 --epochs 3 --results-file results/slim-t5-new-results-ratio125.jsonl
#    rm -rf my_awesome_model
# done



# for i in {1..5}
# do
# #    # full dataset
#    python3 main.py --model t5 --dataset-path slim-dataset-v2.csv.gz --fold-index $i --run-complete-dataset true --ratio 0 --batch-size 16 --epochs 3 --results-file results/slim-t5-results-ratio100.jsonl
#    rm -rf my_awesome_model
# done


# ############## LLAMA MODELS

# for i in {1..5}
# do
#      # ratio 10
#    python3 main.py --model llama2 --dataset-path slim-dataset-v2.csv.gz --fold-index $i --run-complete-dataset false --ratio 10 --batch-size 32 --epochs 3 --results-file results/slim-llama2-results-ratio10.jsonl
#    rm -rf my_awesome_model
# done

# for i in {1..5}
# do
#      # ratio 1 MISTRAL
#    python3 main.py --model mistral --dataset-path slim-dataset-v2.csv.gz --fold-index $i --run-complete-dataset false --ratio 1 --batch-size 32 --epochs 3 --results-file results/slim-mistral-results-ratio1.jsonl
#    rm -rf my_awesome_model
# done
# 
# for i in {3..5}
# do
#      # ratio 1
#    python3 main.py --model llama2 --dataset-path slim-dataset-v2.csv.gz --fold-index $i --run-complete-dataset false --ratio 1 --batch-size 32 --epochs 3 --results-file results/slim-llama2-results-ratio1.jsonl
#    rm -rf my_awesome_model
# done

for i in {2..5}
do
     # ratio 100
   python3 main.py --model llama2 --dataset-path slim-dataset-v2.csv.gz --fold-index $i --run-complete-dataset false --ratio 100 --batch-size 32 --epochs 1 --results-file results/slim-llama2-results-ratio100.jsonl
   rm -rf my_awesome_model
done

#### JOURNALIST VERSION

# for i in {1..5}
# do
#      # ratio 1
#    python3 main.py --model bert --dataset-path slim-journalist-version.zip --fold-index $i --run-complete-dataset false --ratio 1 --batch-size 32 --epochs 3 --results-file results/slim-journalist-bert-results-ratio1.jsonl
#    rm -rf my_awesome_model
# done

# for i in {1..5}
# do
# #    # ratio 10
#    python3 main.py --model bert --dataset-path slim-journalist-version.zip --fold-index $i --run-complete-dataset false --ratio 10 --batch-size 32 --epochs 3 --results-file results/slim-journalist-bert-results-ratio10.jsonl
#    rm -rf my_awesome_model
# done

# for i in {1..5}
# do
# #    # ratio 125
#    python3 main.py --model bert --dataset-path slim-journalist-version.zip --fold-index $i --run-complete-dataset false --ratio 125 --batch-size 32 --epochs 3 --results-file results/slim-journalist-bert-results-ratio125.jsonl
#    rm -rf my_awesome_model
# done

# for i in {1..5}
# do
#      # ratio 1
#    python3 main.py --model t5 --dataset-path slim-journalist-version.zip --fold-index $i --run-complete-dataset false --ratio 1 --batch-size 32 --epochs 3 --results-file results/slim-journalist-t5-results-ratio1.jsonl
#    rm -rf my_awesome_model
# done

# for i in {1..5}
# do
# #    # ratio 10
#    python3 main.py --model t5 --dataset-path slim-journalist-version.zip --fold-index $i --run-complete-dataset false --ratio 10 --batch-size 32 --epochs 3 --results-file results/slim-journalist-t5-results-ratio10.jsonl
#    rm -rf my_awesome_model
# done

# for i in {5..5}
# do
# #    # ratio 10
#    python3 main.py --model t5 --dataset-path slim-journalist-version.zip --fold-index $i --run-complete-dataset false --ratio 100 --batch-size 32 --epochs 3 --results-file results/slim-journalist-t5-results-ratio100.jsonl
#    rm -rf my_awesome_model
# done

# for i in {1..5}
# do
# #    # ratio 125
#    python3 main.py --model t5 --dataset-path slim-journalist-version.zip --fold-index $i --run-complete-dataset false --ratio 125 --batch-size 32 --epochs 3 --results-file results/slim-journalist-t5-results-ratio125.jsonl
#    rm -rf my_awesome_model
# done


# ############## LLAMA MODELS

# for i in {1..5}
# do # ratio 100
#    python3 main.py --model llama --dataset-path slim-dataset-v2.csv.gz --fold-index $i --run-complete-dataset false --ratio 100 --batch-size 32 --epochs 3 --results-file results/slim-llama-results-ratio100.jsonl
#    rm -rf my_awesome_model
# done

# for i in {1..5}
# do
#      # ratio 10
#    python3 main.py --model llama --dataset-path slim-dataset-v2.csv.gz --fold-index $i --run-complete-dataset false --ratio 10 --batch-size 32 --epochs 3 --results-file results/slim-llama-results-ratio10.jsonl
#    rm -rf my_awesome_model
# done

#for i in {2..5}
#do
#     # ratio 1
#   python3 main.py --model llama --dataset-path slim-journalist-version.zip --fold-index $i --run-complete-dataset false --ratio 1 --batch-size 32 --epochs 3 --results-file results/slim-journalist-llama-results-ratio1.jsonl
#   rm -rf my_awesome_model
#done
