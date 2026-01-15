import torch
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from transformers import TrainingArguments, Trainer
from peft import PeftModel
from datasets import Dataset, DatasetDict
from transformers import AdamW, get_scheduler, AutoTokenizer, AutoModelForSequenceClassification

from transformers import BitsAndBytesConfig


nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)


torch.manual_seed(23)

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)

CUTOFF_LEN = 256
BASE_MODEL = "mistralai/Mistral-7B-v0.1"#"meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

tokenizer.pad_token_id = (
    0  # unk. we want this to be different from the eos token
)
tokenizer.padding_side = "left"


data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)

def get_llama():
    BASE_MODEL = "mistralai/Mistral-7B-v0.1"
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT= 0.05
    LORA_TARGET_MODULES = [
        "q_proj",
        "v_proj",
    ]

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=nf4_config
    )

    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = 0

    # model = PeftModel.from_pretrained(model, "22h/cabrita-lora-v0-1")

    model = prepare_model_for_int8_training(model)
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="SEQ_CLS"
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model

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

def generate_prompt(data_point):
    return f"""Abaixo está uma instrução que descreve uma tarefa, juntamente com uma entrada que fornece mais contexto. Escreva uma resposta que complete adequadamente o pedido.
### Instrução:Classifique o seguinte texto como verdadeiro ou falso:
### Entrada:
{data_point}
### Resposta:
"""

def tokenize(prompt, data_point, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < CUTOFF_LEN
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
 
    result["label"] = data_point["label"]
    result["prompt"] = prompt

    return result

def preprocess_function(examples):
    full_prompt = generate_prompt(examples["transcription_text"])
    tokenized_full_prompt = tokenize(full_prompt, examples)
    return tokenized_full_prompt

def train_model(train_data, val_data, test_data, epochs, batch_size, eval_batch_size = 16):
  dataset = DatasetDict({
    "train":Dataset.from_pandas(train_data).map(preprocess_function),
    "validation":Dataset.from_pandas(val_data).map(preprocess_function),
    "test":Dataset.from_pandas(test_data).map(preprocess_function),
  })

  ## parameters
  batch_size = batch_size
  num_epochs = epochs
  micro_batch_size = 4
  gradient_accumulation_steps = batch_size // micro_batch_size

  training_args = TrainingArguments(
      output_dir="my_awesome_model",
      learning_rate=3e-5,
      per_device_train_batch_size=micro_batch_size,
      gradient_accumulation_steps=gradient_accumulation_steps,
      per_device_eval_batch_size=eval_batch_size,
      num_train_epochs=num_epochs,
      weight_decay=0.001,
      evaluation_strategy="epoch",
      do_eval=True,
      save_strategy="epoch",
      load_best_model_at_end=True,
      max_grad_norm=0.3,
  )

  batches_per_epoch = len(dataset["train"]) // batch_size
  total_train_steps = int(batches_per_epoch * num_epochs)

  model = get_llama()

  optimizer = AdamW(model.parameters(), lr=3e-5)

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

  model.config.use_cache = False
  old_state_dict = model.state_dict
  model.state_dict = (
      lambda self, *_, **__: get_peft_model_state_dict(
          self, old_state_dict()
      )
  ).__get__(model, type(model))

  model = torch.compile(model)

  with torch.autocast("cuda"):
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
