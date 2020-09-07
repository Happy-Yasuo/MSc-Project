# -*- coding: utf-8 -*-
import json
import random
import os
import zipfile
from transformers import Trainer, TrainingArguments, RobertaTokenizer, RobertaForSequenceClassification
from transformers.file_utils import cached_path
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch
from nlp import Dataset
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(root_dir)
save_dir = os.path.join(root_dir, 'convlab2/policy/dqn/NLE/save/personachat')
log_dir = os.path.join(root_dir, 'convlab2/policy/dqn/NLE/log')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
url = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"
# Download and load JSON dataset
personachat_file = cached_path(url)
with open(personachat_file, "r", encoding="utf-8") as f:
    dataset = json.loads(f.read())




def artificial(real_usr, real_sys):
    artificial_usr_utter = []
    artificial_sys_utter = []
    num_turn = len(real_usr)
    for i in range(num_turn):
        artificial_usr_utter.append(real_usr[i])
        random_index = random.choice(list(range(i)) + list(range(i + 1, num_turn)))
        artificial_sys_utter.append(real_sys[random_index])
    usr_utter = real_usr + artificial_usr_utter
    sys_utter = real_sys + artificial_sys_utter
    return usr_utter, sys_utter


real_train_usr_utter = []
real_train_sys_utter = []
for item in dataset['train']:
    for ind, turn in enumerate(item['utterances'][-1]['history']):
        if ind % 2 == 0:
            real_train_usr_utter.append(turn)
        else:
            real_train_sys_utter.append(turn)
    if len(real_train_usr_utter) > len(real_train_sys_utter):
        real_train_usr_utter.pop()

real_eval_usr_utter = []
real_eval_sys_utter = []
for item in dataset['valid']:
    for ind, turn in enumerate(item['utterances'][-1]['history']):
        if ind % 2 == 0:
            real_eval_usr_utter.append(turn)
        else:
            real_eval_sys_utter.append(turn)
    if len(real_eval_usr_utter) > len(real_eval_sys_utter):
        real_eval_usr_utter.pop()

train_usr_utter, train_sys_utter = artificial(real_train_usr_utter, real_train_sys_utter)
val_usr_utter, val_sys_utter = artificial(real_eval_usr_utter, real_eval_sys_utter)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained('roberta-base')
train_encoding = tokenizer(train_usr_utter, train_sys_utter, padding=True, truncation=True, max_length=100)
train_encoding['label'] = [1] * (len(train_usr_utter)//2) + [0] * (len(train_usr_utter)//2)
train_dataset = Dataset.from_dict(train_encoding)
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

val_encoding = tokenizer(val_usr_utter, val_sys_utter, padding=True, truncation=True, max_length=100)
val_encoding['label'] = [1] * (len(val_usr_utter)//2) + [0] * (len(val_usr_utter)//2)
val_dataset = Dataset.from_dict(val_encoding)
val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


training_args = TrainingArguments(
    output_dir=save_dir,
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=128,
    warmup_steps=500,
    weight_decay=0.01,
    evaluate_during_training=True,
    logging_dir=log_dir,
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()
evaluate_output = trainer.evaluate()
print(evaluate_output)

trainer.save_model(os.path.join(save_dir, 'personachat_checkpoint'))

