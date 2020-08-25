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
save_dir = os.path.join(root_dir, 'convlab2/policy/dqn/NLE/save/script')
log_dir = os.path.join(root_dir, 'convlab2/policy/dqn/NLE/log')
script_dir = os.path.join(root_dir, 'convlab2/policy/dqn/NLE/script')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

frame_1 = os.path.join(script_dir, 'frame_1.txt')
frame_2 = os.path.join(script_dir, 'frame_2.txt')

with open(frame_1, 'r') as f1:
    first_frame = []
    for line in f1.readlines():
        first_frame.append(line.strip())

with open(frame_2, 'r') as f2:
    second_frame = []
    for line in f2.readlines():
        second_frame.append(line.strip())

# print("Length of frame 1：{length}".format(length=len(first_frame)))
# print("Length of frame 2：{length}".format(length=len(second_frame)))
# for i in range(20, 10, -1):
#     print(first_frame[-i])
#     print(second_frame[-i])


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


real_eval_usr_utter = []
real_eval_sys_utter = []

pos = 4
while pos < len(first_frame):
    real_eval_usr_utter.append(first_frame.pop(pos))
    real_eval_sys_utter.append(second_frame.pop(pos))
    pos += 4

real_train_usr_utter = first_frame
real_train_sys_utter = second_frame

train_usr_utter, train_sys_utter = artificial(real_train_usr_utter, real_train_sys_utter)
val_usr_utter, val_sys_utter = artificial(real_eval_usr_utter, real_eval_sys_utter)

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

trainer.save_model(os.path.join(save_dir, 'script_checkpoint'))

