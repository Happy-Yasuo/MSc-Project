# -*- coding: utf-8 -*-
import json
import random
import os
import zipfile
from transformers import Trainer, TrainingArguments, RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch
from nlp import Dataset
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(root_dir)
save_dir = os.path.join(root_dir, 'convlab2/policy/dqn/NLE/save/taskmaster')
log_dir = os.path.join(root_dir, 'convlab2/policy/dqn/NLE/log')
taskmaster_dir = os.path.join(root_dir, 'convlab2/policy/dqn/NLE/Taskmaster')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_file_1 = os.path.join(taskmaster_dir, 'TM-1-2019/woz-dialogs.json')
train_file_2 = os.path.join(taskmaster_dir, 'TM-2-2020/data/flights.json')
train_file_3 = os.path.join(taskmaster_dir, 'TM-2-2020/data/food-ordering.json')
train_file_4 = os.path.join(taskmaster_dir, 'TM-2-2020/data/movies.json')
train_file_5 = os.path.join(taskmaster_dir, 'TM-2-2020/data/music.json')
train_file_6 = os.path.join(taskmaster_dir, 'TM-2-2020/data/sports.json')


with open(train_file_1, 'r') as f1:
    woz_dialogs = json.load(f1)

with open(train_file_2, 'r') as f2:
    flights = json.load(f2)

with open(train_file_3, 'r') as f3:
    food_ordering = json.load(f3)

with open(train_file_4, 'r') as f4:
    movies = json.load(f4)

with open(train_file_5, 'r') as f5:
    music = json.load(f5)

with open(train_file_6, 'r') as f6:
    sports = json.load(f6)


def extract_data(originial_data, exclude_list=None):
    real_usr_utter = []
    real_sys_utter = []
    eval_usr_utter = []
    eval_sys_utter = []
    for no, sess in enumerate(originial_data):
        if exclude_list != None and sess['instruction_id'] in exclude_list:
            continue
        speaker = None
        cur_sys_utter = []
        cur_usr_utter = []
        for id, item in enumerate(originial_data[no]['utterances']):

            if item['speaker'] == 'ASSISTANT':
                if not cur_sys_utter and id == 0:
                    continue
                elif speaker == 'ASSISTANT':
                    cur_sys_utter.append(item['text'])
                elif speaker == 'USER':
                    if not cur_sys_utter:
                        cur_sys_utter.append(item['text'])
                    else:
                        cur_sys_utter[-1] = cur_sys_utter[-1] + ' ' + item['text']
                speaker = 'USER'
            if item['speaker'] == 'USER':
                if not cur_usr_utter:
                    cur_usr_utter.append(item['text'])
                elif speaker == 'USER':
                    cur_usr_utter.append(item['text'])
                else:
                    cur_usr_utter[-1] = cur_usr_utter[-1] + ' ' + item['text']
                speaker = 'ASSISTANT'
        if len(cur_usr_utter) < len(cur_sys_utter):
            cur_sys_utter.pop()
        if len(cur_usr_utter) > len(cur_sys_utter):
            cur_usr_utter.pop()

        real_sys_utter += cur_sys_utter
        real_usr_utter += cur_usr_utter

    pos = 4
    while pos < len(real_usr_utter):
        eval_usr_utter.append(real_usr_utter.pop(pos))
        eval_sys_utter.append(real_sys_utter.pop(pos))
        pos += 4

    return real_usr_utter, real_sys_utter, eval_usr_utter, eval_sys_utter


data = [woz_dialogs, flights, food_ordering, movies, sports]


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


def generate_data(data):
    exclude = ['restaurant-table-1', 'restaurant-table-2', 'uber-lyft-1', 'uber-lyft-2']

    real_train_usr_utter = []
    real_train_sys_utter = []
    real_eval_usr_utter = []
    real_eval_sys_utter = []
    for source_data in data:
        if source_data == woz_dialogs:
            cur_usr_1, cur_sys_1, cur_usr_2, cur_sys_2 = extract_data(source_data, exclude)
        else:
            cur_usr_1, cur_sys_1, cur_usr_2, cur_sys_2 = extract_data(source_data)
        real_train_usr_utter += cur_usr_1
        real_train_sys_utter += cur_sys_1
        real_eval_usr_utter += cur_usr_2
        real_eval_sys_utter += cur_sys_2

    train_usr, train_sys = artificial(real_train_usr_utter, real_train_sys_utter)
    eval_usr, eval_sys = artificial(real_eval_usr_utter, real_eval_sys_utter)
    return train_usr, train_sys, eval_usr, eval_sys


train_usr_utter, train_sys_utter, val_usr_utter, val_sys_utter = generate_data(data)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained('roberta-base')
train_encoding = tokenizer(train_usr_utter, train_sys_utter, padding=True, truncation=True, max_length=80)
train_encoding['label'] = [1] * (len(train_usr_utter)//2) + [0] * (len(train_usr_utter)//2)
train_dataset = Dataset.from_dict(train_encoding)
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
val_encoding = tokenizer(val_usr_utter, val_sys_utter, padding=True, truncation=True, max_length=80)
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
    num_train_epochs=2,
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

trainer.save_model(os.path.join(save_dir, 'taskmaster_checkpoint'))
