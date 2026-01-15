# python3 main.py --full-dataset complete-dataset.csv --fold-index=4 --ratio 10 --model bert --batch-size 64 --epochs 3

import argparse
import pandas as pd
import json
import time
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import random
from bertimbau import train_model as bertimbau_train_model
from ptt5 import train_model as ptt5_train_model
# from llama import train_model as llama_train_model
from llama2 import train_model as llama2_train_model
from mistral import train_model as mistral_train_model
import torch 

torch.manual_seed(23)
parser = argparse.ArgumentParser()
# Adding optional argument
parser.add_argument("--dataset-path", help = "Path to a csv file containing all instances")
parser.add_argument("--fold-index", help = "Index of fold for cross validation. Ranges from 1 to 5")
parser.add_argument("--run-complete-dataset", help = "A flag to signal that we wont use the undersampling ratio option")
parser.add_argument("--ratio", help = "Ratio of positive-to-negative examples")
parser.add_argument("--model", help = "which model to run: bert, t5, llama or llama2")
parser.add_argument("--batch-size", help = "batch size")
parser.add_argument("--epochs", help = "numer of epochs")
parser.add_argument("--results-file", help = "file to store results")
parser.add_argument("--oversample-class1", help = "Factor by which to oversample class 1; '0' means not oversampling", nargs='?', const=0, type=int, required=False)
parser.add_argument("--validation-ratio", help = "Set the rate of positive to negative examples in the validation set. Default is 1 to 1", nargs='?', const=1, type=int, required=False)
# Read arguments from command line
args = parser.parse_args()
 
dataset_path = args.dataset_path
fold_index = int(args.fold_index)

if args.run_complete_dataset == "true":
    run_complete_dataset = 1
else: 
   run_complete_dataset = 0

undersample_ratio = 0
if run_complete_dataset == 0: #only parses undesample if we wont use the full dataset
   undersample_ratio = int(args.ratio)


print("RATIO", undersample_ratio)
model = args.model
batch_size = int(args.batch_size)
epochs = int(args.epochs)
results_filename = args.results_file
oversample_class1 = int(args.oversample_class1)
validation_ratio = int(args.validation_ratio)


df = pd.read_csv(dataset_path, dtype={'transcription_index': str})
files = list(df["file"].unique())


kf = KFold(n_splits=5, shuffle=True, random_state=23)
fold_counter = 1
for train_index, test_index in kf.split(files):
    if(fold_counter==fold_index):
        print("FOLD", fold_counter)       
        aux_train_files = [files[i] for i in train_index]
        # train_test_split returns the items themselves, not indexes
        train_files ,val_files = train_test_split(aux_train_files,test_size=0.25, random_state = 23)
        test_files = [files[j] for j in test_index]

        train = df[df["file"].isin(train_files)]
        val = df[df["file"].isin(val_files)]
        test = df[df["file"].isin(test_files)]

        #shuffling to decrease biases
        train = train.sample(frac=1, random_state=23)
        test = test.sample(frac=1, random_state=23)
        val = val.sample(frac=1, random_state=23)


        if run_complete_dataset == 0:
            train_positive = train[train["label"]==1]
            train_negative = train[train["label"]==0]
            try:
                train_negative = train_negative.sample(len(train_positive) * undersample_ratio, random_state=23)
                if(oversample_class1 > 0):
                    print("ATTENTION: oversampling by a factor of",oversample_class1)
                    train_positive = train_positive.sample(frac=oversample_class1, replace=True, random_state=23)
                    print("positive examples:", len(train_positive))

            except Exception as e:
                # if we want to undesample more than we can, exit
                print("entrou no except:::::::")
                print(len(train_positive) * undersample_ratio, len(train)+len(val))

                exit()

            train = pd.concat([train_positive,train_negative])

        val_positive = val[val["label"]==1]
        val_negative = val[val["label"]==0]
        # avaliar/revisar
        val_negative = val_negative.sample(len(val_positive) * validation_ratio, random_state=23)
        validation = pd.concat([val_positive,val_negative])
        
        train = train.sample(frac=1,random_state=23)

        validation = validation.sample(frac=1, random_state=23)
        test = test.sample(frac=1,random_state=23)       

        print("TREINO", len(train))
        print("VALIDACAO", len(validation)) 
        print("TESTE", len(test))        

        start = time.time()
        if(model=="bert"):
            result = bertimbau_train_model(train, validation, test, batch_size=batch_size,epochs=epochs)
        elif(model=="t5"):
            result = ptt5_train_model(train, validation, test, batch_size=batch_size,epochs=epochs)
        # elif(model=="llama"):
        #     result = llama_train_model(train, validation, test, batch_size=batch_size,epochs=epochs)
        elif(model=="llama2"):
            result = llama2_train_model(train, validation, test, batch_size=batch_size,epochs=epochs)
        elif(model=="mistral"):
            result = mistral_train_model(train, validation, test, batch_size=batch_size,epochs=epochs)
        end = time.time()
        print("TIME IN SECONDS", int(end-start))

        result["test_size"] = len(test)
        result["train_size"] = len(train)
        result["training_time"] = end-start
        result["fold"] = fold_counter
        result["model"] = model
        result["dataset"] = dataset_path
        result["batch_size"] = batch_size
        result["epochs"] = epochs
        result["run_complete_dataset"] = run_complete_dataset
        result["undersample_ratio"] = undersample_ratio
        with open(results_filename, "a+") as f:
            f.write(json.dumps(result)+"\n")
        print("CORRECT FOLD_INDEX:", fold_counter)
    else:
        pass
    fold_counter += 1
