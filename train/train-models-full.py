# python3 train-models-full.py --output-dir bert-full --push-to-hub 0 --model bert --dataset-path slim-dataset-v2.csv.gz --ratio 75 --batch-size 32 --epochs 3 --test-dataset-path updated-individual-instances.csv
import argparse
import pandas as pd
import json
import evaluate
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AdamW, get_scheduler, AutoTokenizer, DataCollatorWithPadding
import torch
from sklearn.model_selection import train_test_split
import tqdm
from functools import partial


torch.manual_seed(23)

def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    weighted_accuracy = evaluate.load("accuracy", average="weighted")
    precision = evaluate.load("precision")
    weighted_precision = evaluate.load("precision", average="weighted")
    f1 = evaluate.load("f1")
    weighted_f1 = evaluate.load("f1", average="weighted")
    macrof1 = evaluate.load('f1', average='macro')
    microf1 = evaluate.load('f1', average='micro')
    recall = evaluate.load("recall")
    weighted_recall = evaluate.load("recall", average="weighted")

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    computed_acc = accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    computed_balanced_acc = balanced_accuracy_score(y_true = labels, y_pred = predictions)
    computed_precision = precision.compute(predictions=predictions, references=labels)["precision"]
    computed_f1 = f1.compute(predictions=predictions, references=labels)["f1"]
    computed_macrof1 = macrof1.compute(predictions=predictions, references=labels, average="macro")["f1"]
    computed_microf1 = microf1.compute(predictions=predictions, references=labels, average="micro")["f1"]
    computed_recall = recall.compute(predictions=predictions, references=labels)["recall"]

    computed_weighted_precision = weighted_precision.compute(predictions=predictions, references=labels, average="weighted")["precision"]
    computed_weighted_precision_class0 = weighted_precision.compute(predictions=predictions, references=labels, average="weighted", labels=[0])["precision"]
    computed_weighted_precision_class1 = weighted_precision.compute(predictions=predictions, references=labels, average="weighted", labels=[1])["precision"]


    computed_weighted_f1 = weighted_f1.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    computed_weighted_f1_class0 = weighted_f1.compute(predictions=predictions, references=labels, average="weighted", labels=[0])["f1"]
    computed_weighted_f1_class1 = weighted_f1.compute(predictions=predictions, references=labels, average="weighted", labels=[1])["f1"]

    computed_weighted_recall = weighted_recall.compute(predictions=predictions, references=labels, average="weighted")["recall"]
    computed_weighted_recall_class0 = weighted_recall.compute(predictions=predictions, references=labels, average="weighted", labels=[0])["recall"]
    computed_weighted_recall_class1 = weighted_recall.compute(predictions=predictions, references=labels, average="weighted", labels=[1])["recall"]


    return {"accuracy": computed_acc,
            "balanced_accuracy": computed_balanced_acc,
            "precision": computed_precision,
            "recall": computed_recall,
            "f1": computed_f1,
            "macrof1": computed_macrof1,
            "microf1": computed_microf1,
            "weighted_precision": computed_weighted_precision,
            "weighted_precision_class0":computed_weighted_precision_class0,
            "weighted_precision_class1":computed_weighted_precision_class1,
            "weighted_f1": computed_weighted_f1,
            "weighted_f1_class0":computed_weighted_f1_class0,
            "weighted_f1_class1":computed_weighted_f1_class1,
            "weighted_recall": computed_weighted_recall,
            "weighted_recall_class0":computed_weighted_recall_class0,
            "weighted_recall_class1":computed_weighted_recall_class1,
            }


def preprocess_function(examples, tokenizer):
    return tokenizer(examples["transcription_text"], truncation=True)

def train_bertimbau(train_data, validation_data, epochs, batch_size, output_dir, input_params, push_to_hub=0, opt_suffix = "", test_dataset = None):
  model_path = 'neuralmind/bert-base-portuguese-cased'
  tokenizer = AutoTokenizer.from_pretrained(model_path,model_max_length=256)
  data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
  mapfunc = partial(preprocess_function, tokenizer=tokenizer)
  dataset = DatasetDict({
    "train":Dataset.from_pandas(train_data).map(mapfunc, batched=True),
    "validation":Dataset.from_pandas(validation_data).map(mapfunc, batched=True)
  })
  ## parameters
  batch_size = batch_size
  num_epochs = epochs
  micro_batch_size = 32
  eval_batch_size = 64
  gradient_accumulation_steps = batch_size // micro_batch_size

  training_args = TrainingArguments(
      output_dir=output_dir,
      learning_rate=2e-5,
      per_device_train_batch_size=micro_batch_size,
      gradient_accumulation_steps=gradient_accumulation_steps,
      per_device_eval_batch_size=eval_batch_size,
      num_train_epochs=num_epochs,
      weight_decay=0.01,
      evaluation_strategy="epoch",
      do_eval=True,
      save_strategy="epoch",
      load_best_model_at_end=True,
  )

  batches_per_epoch = len(dataset["train"]) // batch_size
  total_train_steps = int(batches_per_epoch * num_epochs)

  model = AutoModelForSequenceClassification.from_pretrained(
      model_path, num_labels=2
  )

  optimizer = AdamW(model.parameters(), lr=5e-5)

  num_training_steps = total_train_steps
  lr_scheduler = get_scheduler(
      "linear",
      optimizer=optimizer,
      num_warmup_steps=int(0.1 * total_train_steps),
      num_training_steps=num_training_steps,
  )

  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=dataset["train"],
      eval_dataset=dataset["validation"],
      tokenizer=tokenizer,
      data_collator=data_collator,
      optimizers=[optimizer, lr_scheduler],
      compute_metrics=compute_metrics,
  )

  trainer.train()

  if(test_dataset is not None):
    test = DatasetDict({
    "test":Dataset.from_pandas(test_dataset).map(mapfunc, batched=True)
    })
    predictions, _, metrics = trainer.predict(test["test"])
    predictions = predictions.tolist()

    results_dict = {
        "metrics": dict(metrics),
        "training_params_parameters": input_params,
        "predictions": list(predictions),
        "predictions_argmax": np.argmax(list(predictions), axis=1).tolist(),
        "video_id": list(test["test"]["file"]),
        "sentence_id": list(test["test"]["transcription_index"]),
        "label": list(test["test"]["label"]),
        "fact-check-id": list(test["test"]["fact_check_id"]),
    }

    with open("results/bertimbau-fullmodel-result-ratio"+str(input_params['ratio'])+".jsonl", "w") as f:
       f.write(json.dumps(results_dict))

  if(push_to_hub == 1):
    print("brenomatos/"+model_path.split("/")[-1]+opt_suffix)
    trainer.push_to_hub("brenomatos/"+model_path.split("/")[-1]+opt_suffix)


def train_ptt5(train_data, validation_data, epochs, batch_size, output_dir, push_to_hub=0, opt_suffix = "", test_dataset = None):
    model_path = 'unicamp-dl/ptt5-base-portuguese-vocab'
    tokenizer = AutoTokenizer.from_pretrained(model_path,model_max_length=256)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    mapfunc = partial(preprocess_function, tokenizer=tokenizer)
    
    dataset = DatasetDict({
    "train":Dataset.from_pandas(train_data).map(preprocess_function, batched=True),
    "validation":Dataset.from_pandas(validation_data).map(preprocess_function, batched=True),
  })

    # parameters
    batch_size = batch_size
    num_epochs = epochs
    micro_batch_size = 8
    gradient_accumulation_steps = batch_size // micro_batch_size

    training_args = TrainingArguments(
        output_dir="my_awesome_model",
        learning_rate=1e-4,
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        do_eval=True,
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    batches_per_epoch = len(dataset["train"]) // batch_size
    total_train_steps = int(batches_per_epoch * num_epochs)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=2
    )

    optimizer = AdamW(model.parameters())

    num_training_steps = total_train_steps
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * total_train_steps),
        num_training_steps=num_training_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=[optimizer, lr_scheduler],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    if(test_dataset is not None):
        test = DatasetDict({
        "test":Dataset.from_pandas(test_dataset).map(mapfunc, batched=True)
        })
        predictions = []
        predictions.append([])
        predictions.append([])
        
        with torch.no_grad():
            for i in tqdm.tqdm(range((len(dataset["test"])//1000)+1)):
                aux_predictions, _, aux_metrics = trainer.predict(dataset["test"].select(range(i*1000,min(len(dataset["test"]),(i+1)*1000))))
                predictions[0].extend(aux_predictions[0])
                predictions[1].extend(aux_predictions[1])

        results_dict = {
            "metrics": dict(compute_metrics((predictions, list(dataset["test"]["label"])))),
            #   "predictions": list(predictions[0]),
            "predictions_argmax": np.argmax(list(predictions[0]), axis=1).tolist(),
            "video_id": list(dataset["test"]["file"]),
            "sentence_id": list(dataset["test"]["transcription_index"]),
            "label": list(dataset["test"]["label"]),
            "fact-check-id": list(dataset["test"]["fact_check_id"])
        }

        with open("ptt5-fullmodel-result.jsonl", "w") as f:
            f.write(json.dumps(results_dict))

    if(push_to_hub):  
        trainer.push_to_hub("brenomatos/"+model_path.split("/")[-1]+opt_suffix)

def main():
    parser = argparse.ArgumentParser()
    # Adding optional argument
    parser.add_argument("--dataset-path", help = "Path to a csv file containing all instances")
    parser.add_argument("--ratio", help = "Ratio of positive-to-negative examples")
    parser.add_argument("--model", help = "which model to run: bert, t5, llama or llama2")
    parser.add_argument("--batch-size", help = "batch size")
    parser.add_argument("--epochs", help = "numer of epochs")
    parser.add_argument("--output-dir", help = "local path to save the models")
    parser.add_argument("--push-to-hub", default=0,help = "1 for yes, 0 for no; pushes to your hf hub (must be logged in)")
    parser.add_argument("--optional-suffix", help = "Optional suffix to add to the name of the model pushed to hf")
    parser.add_argument("--test-dataset-path", help = "path for the test dataset")

    args = parser.parse_args()

    dataset_path = args.dataset_path
    undersample_ratio = int(args.ratio)
    model = args.model
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    output_dir = args.output_dir
    push_to_hub = int(args.push_to_hub)
    optional_suffix = args.optional_suffix
    test_dataset = args.test_dataset_path
    input_params = vars(args)

    if(test_dataset is not None):
        test_dataset = pd.read_csv(args.test_dataset_path, dtype={'transcription_index': str}).sample(frac=1, random_state=23)
    
    if(optional_suffix is  None):
        optional_suffix = ""

    print("RATIO", undersample_ratio)
    print(args)

    # undersampling
    df = pd.read_csv(dataset_path, dtype={'transcription_index': str})
    files = list(df["file"].unique())

    train_files ,val_files = train_test_split(files,test_size=0.20, random_state = 23)

    train = df[df["file"].isin(train_files)]
    val = df[df["file"].isin(val_files)]

    train = train.sample(frac=1, random_state=23)
    val = val.sample(frac=1, random_state=23)

    train_positive = train[train["label"]==1]
    train_negative = train[train["label"]==0]
    try:
        train_negative = train_negative.sample(len(train_positive) * undersample_ratio, random_state=23)
    except Exception as e:
        # if we want to undesample more than we can, exit
        print("you want to undesample more than we can, exit:::::::")
        print(len(train_positive) * undersample_ratio)
        exit()

    train = pd.concat([train_positive,train_negative])

    val_positive = val[val["label"]==1]
    val_negative = val[val["label"]==0]
    # avaliar/revisar
    val_negative = val_negative.sample(len(val_positive) * 1, random_state=23)
    validation = pd.concat([val_positive,val_negative])

    print("TRAIN SIZE:", len(train))
    print("VALIDATION SIZE:", len(validation))

    if(model=="bert"):
        train_bertimbau(train, validation, epochs, batch_size, output_dir, input_params, push_to_hub, optional_suffix, test_dataset)


############ old 
#    train = pd.read_csv(dataset_path, dtype={'transcription_index': str})
#    train = train.sample(frac=1, random_state=23)
#
#    train_positive = train[train["label"]==1]
#    train_negative = train[train["label"]==0]
#    try:
#        train_negative = train_negative.sample(len(train_positive) * undersample_ratio, random_state=23)
#    except Exception as e:
#        # if we want to undesample more than we can, exit
#        print("you want to undesample more than we can, exit:::::::")
#        print(len(train_positive) * undersample_ratio)
#        exit()
#
#    train = pd.concat([train_positive,train_negative])
#    print("TRAIN SIZE:", len(train))
#
#    if(model=="bert"):
#        train_bertimbau(train,epochs, batch_size, output_dir, push_to_hub, optional_suffix, test_dataset)

if __name__ == "__main__":
    main()

