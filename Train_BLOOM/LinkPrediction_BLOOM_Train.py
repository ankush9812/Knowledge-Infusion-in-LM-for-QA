import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer
from transformers import BloomTokenizerFast, BloomForCausalLM
from transformers import EarlyStoppingCallback
import random
import evaluate

path_to_triplets="../data/CWQ_triplets" # CHANGE PATH TO DIRECTORT OF KG TRIPLETS

# Read data
with open(path_to_triplets+"/train.txt", 'r') as fp:
    train = fp.read().split('\n')
with open(path_to_triplets+"/valid.txt", 'r') as fp:
    val = fp.read().split('\n')
with open(path_to_triplets+"/test.txt", 'r') as fp:
    test = fp.read().split('\n')

model_name = "bigscience/bloom-1b7"	# CHANGE BLOOM MODEL NAME
tokenizer = BloomTokenizerFast.from_pretrained(model_name)
model = BloomForCausalLM.from_pretrained(model_name)

# ----- 1. Preprocess data -----#
# Preprocess data

train_tokenized = tokenizer(train, padding=True, truncation=True, max_length=100)
val_tokenized = tokenizer(val, padding=True, truncation=True, max_length=100)

# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

train_dataset = Dataset(train_tokenized)
val_dataset = Dataset(val_tokenized)

# ----- 2. Fine-tune pretrained model -----#
# Define Trainer parameters
def compute_metrics(p):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = p
    predictions = np.argmax(logits, axis=-1)
    return {"eval_loss":metric.compute(predictions=predictions, references=labels)}

# Define Trainer
args = TrainingArguments(
	# SET HYPER PARAMETERS
    output_dir="CWQ_output_Triplet_32", # SET PATH WHERE YOU WISH TO SAVE THE FINETUNED MODEL
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    seed=0,
    # load_best_model_at_end=True,
    include_inputs_for_metrics=True,
    fp16=True,
)
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels=inputs["input_ids"]
        # forward pass
        outputs = model(**inputs,labels=labels)
        return (outputs.loss, outputs) if return_outputs else outputs.loss
trainer = CustomTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train pre-trained model
trainer.train()
