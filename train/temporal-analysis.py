import argparse
import pandas as pd
import json
import time
import time
import shutil
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from tqdm import tqdm
tqdm.pandas()
from bertimbau import train_model as bertimbau_train_model
from ptt5 import train_model as ptt5_train_model
from llama import train_model as llama_train_model

parser = argparse.ArgumentParser()
# Adding optional argument
parser.add_argument("--dataset-path", help = "Path to a csv file containing all instances")
parser.add_argument("--fold-index", help = "Index of fold for cross validation. Ranges from 1 to 5")
parser.add_argument("--sliding-type", help = "Should be 'expand-test' (train set increases) or 'walk-forward' (fixed training set length)")
parser.add_argument("--sliding-step", help = "How many months to slide every iteration")
parser.add_argument("--test-length", help = "Number of months to be used in the TEST set")
parser.add_argument("--train-length", help = "Number of months to be used in the TRAIN set. Should be used only wh")
parser.add_argument("--ratio", help = "Ratio of positive-to-negative examples")
parser.add_argument("--model", help = "which model to run")
parser.add_argument("--batch-size", help = "batch size")
parser.add_argument("--epochs", help = "numer of epochs")
parser.add_argument("--results-file", help = "file to store results")
# Read arguments from command line
args = parser.parse_args()
 
dataset_path = args.dataset_path
undersample_ratio = int(args.ratio)

print("RATIO", undersample_ratio)
model = args.model
batch_size = int(args.batch_size)
epochs = int(args.epochs)
results_filename = args.results_file
sliding_type = args.sliding_type
sliding_step = int(args.sliding_step)
test_length = int(args.test_length)
train_length = int(args.train_length)


df = pd.read_csv(dataset_path)
files = list(df["file"].unique())

df["fixed_date"] = pd.to_datetime(df["fixed_date"])
df["comment-year-month"] = df["fixed_date"].apply(lambda x: str(x.year) + "-" + str(x.month))
df["comment-year-month"]

year_index_dict = {}
counter = 1
for year in range(2019,2023):
  for month in range(1,13):
    year_index_dict[str(year)+"-"+str(month)] = counter
    counter +=1
df["month-index"] = df["comment-year-month"].progress_apply(lambda x: year_index_dict[x])


def walk_forward(data, train_window_size, test_window_size, increment=1, mode="expand-test"):
  for index in range(0,len(data),increment):
    train_start = index
    train_end = index+train_window_size
    test_start = index+train_window_size
    test_end = index+train_window_size+test_window_size
    # this if guarantees that we only train and test splits that have complete sizes (i.e., the test size will always have the correct span)
    if(len(data[test_start:test_end]) == test_window_size):
      print(mode)
      if(mode=="expand-test"):
        yield data[:train_end], data[test_start:test_end]
      if(mode=="walk-forward"):
        yield data[train_start:train_end], data[test_start:test_end]


x = np.arange(len(year_index_dict))

for trn_index, tst_index in walk_forward(data=x,
                                            train_window_size=train_length,
                                            test_window_size=test_length,
                                            increment=sliding_step,
                                            mode=sliding_type):
    train_index = trn_index[:-1]
    val_index = [trn_index[-1]]
    
    val = df[df["month-index"].isin(val_index)]
    val_positive = val[val["label"]==1]
    val_negative = val[val["label"]==0].sample(len(val_positive) * 1,random_state=23)
    validation = pd.concat([val_positive, val_negative])
    test = df[df["month-index"].isin(tst_index)]
    train = df[df["month-index"].isin(trn_index)]
    
    if len(test) and len(validation):
      positive = train[train["label"]==1]
      negative = train[train["label"]==0].sample(len(positive) * undersample_ratio,random_state=23)
      train = pd.concat([positive, negative])
      train = train.sample(frac=1, random_state=23)
      print(len(train), len(validation), validation["comment-year-month"].values[0], len(test), test["comment-year-month"].values[0])
      start = time.time()
      if(model=="bert"):
          result = bertimbau_train_model(train, validation, test, batch_size=batch_size,epochs=epochs)
      elif(model=="t5"):
          result = ptt5_train_model(train, validation, test, batch_size=batch_size,epochs=epochs)
      elif(model=="llama"):
          result = llama_train_model(train, validation, test, batch_size=batch_size,epochs=epochs)
      end = time.time()
      result["test_size"] = len(test)
      result["train_size"] = len(train)
      result["training_time"] = end-start
      result["model"] = model
      result["dataset"] = dataset_path
      result["batch_size"] = batch_size
      result["epochs"] = epochs
      result["undersample_ratio"] = undersample_ratio
      result["sliding_type"] = sliding_type
      result["sliding_step"] = sliding_step
      result["test_length"] = test_length
      result["train_length"] = train_length
      result["test_month"] = test["comment-year-month"].values[0]
      result["experiment_finish_date"] = time.time()

      with open(results_filename, "a+") as f:
          f.write(json.dumps(result)+"\n")
      try:
        shutil.rmtree("my_awesome_model")
      except:
        pass
    else:
      result = {}
      result["test_size"] = len(test)
      result["train_size"] = len(train)
      result["training_time"] = 0
      result["model"] = model
      result["dataset"] = dataset_path
      result["batch_size"] = batch_size
      result["epochs"] = epochs
      result["undersample_ratio"] = undersample_ratio
      result["sliding_type"] = sliding_type
      result["sliding_step"] = sliding_step
      result["test_length"] = test_length
      result["train_length"] = train_length
      result["test_month"] = "null"
      result["experiment_finish_date"] = time.time()

      with open(results_filename, "a+") as f:
          f.write(json.dumps(result)+"\n")