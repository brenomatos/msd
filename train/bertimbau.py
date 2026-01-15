import evaluate
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AdamW, get_scheduler, AutoTokenizer, DataCollatorWithPadding
import torch

torch.manual_seed(23)

model_path = 'neuralmind/bert-base-portuguese-cased'
tokenizer = AutoTokenizer.from_pretrained(model_path,model_max_length=256)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

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

def preprocess_function(examples):
    return tokenizer(examples["transcription_text"], truncation=True)

def train_model(train_data, val_data, test_data, epochs, batch_size, eval_batch_size = 64):

  dataset = DatasetDict({
    "train":Dataset.from_pandas(train_data).map(preprocess_function, batched=True),
    "validation":Dataset.from_pandas(val_data).map(preprocess_function, batched=True),
    "test":Dataset.from_pandas(test_data).map(preprocess_function, batched=True),
  })


  ## parameters
  batch_size = batch_size
  num_epochs = epochs
  micro_batch_size = 32
  gradient_accumulation_steps = batch_size // micro_batch_size

  training_args = TrainingArguments(
      output_dir="my_awesome_model",
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
  predictions, _, metrics = trainer.predict(dataset["test"])
  predictions = predictions.tolist()

  results_dict = {
      "metrics": dict(metrics),
      "predictions": list(predictions),
      "predictions_argmax": np.argmax(list(predictions), axis=1).tolist(),
      "video_id": list(dataset["test"]["file"]),
      "sentence_id": list(dataset["test"]["transcription_index"]),
      "label": list(dataset["test"]["label"]),
      "fact-check-id": list(dataset["test"]["fact_check_id"])
  }

  return results_dict

