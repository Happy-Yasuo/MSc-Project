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
save_dir = os.path.join(root_dir, 'convlab2/policy/dqn/NLE/save')
log_dir = os.path.join(root_dir, 'convlab2/policy/dqn/NLE/logs')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_zipped_json(filepath, filename):
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))


train_file = os.path.join(root_dir, 'data/multiwoz/train.json.zip')
val_file = os.path.join(root_dir, 'data/multiwoz/val.json.zip')
test_file = os.path.join(root_dir, 'data/multiwoz/test.json.zip')
multiwoz_train = read_zipped_json(train_file, 'train.json')
multiwoz_val = read_zipped_json(val_file, 'val.json')
multiwoz_test = read_zipped_json(test_file, 'test.json')


def generate_data(originial_data):
    real_usr_utter = []
    real_sys_utter = []

    for no, sess in originial_data.items():
        for is_sys, turn in enumerate(sess['log']):
            turn['text'] = turn['text'].replace("\n", "")
            turn['text'] = turn['text'].replace("\t", "")
            if is_sys % 2 == 0:
                real_usr_utter.append(turn['text'])
            if is_sys % 2 == 1:
                real_sys_utter.append(turn['text'])
        if is_sys%2 == 0:
            print('Warning: odd number of frames in a conversation')

    # negative samples
    artificial_usr_utter = []
    artificial_sys_utter = []
    num_turn = len(real_usr_utter)
    for i in range(num_turn):
        artificial_usr_utter.append(real_usr_utter[i])
        random_index = random.choice(list(range(i))+list(range(i+1,10)))
        artificial_sys_utter.append(real_sys_utter[random_index])
    usr_utter = real_usr_utter + artificial_usr_utter
    sys_utter = real_sys_utter + artificial_sys_utter
    return usr_utter, sys_utter


train_usr_utter, train_sys_utter = generate_data(multiwoz_train)
val_usr_utter, val_sys_utter = generate_data(multiwoz_val)
test_usr_utter, test_sys_utter = generate_data(multiwoz_test)

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
test_encoding = tokenizer(test_usr_utter, test_sys_utter, padding=True, truncation=True, max_length=80)
test_encoding['label'] = [1] * (len(test_usr_utter)//2) + [0] * (len(test_usr_utter)//2)
test_dataset = Dataset.from_dict(test_encoding)
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])


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

trainer.save_model(os.path.join(save_dir, 'final_checkpoint'))
