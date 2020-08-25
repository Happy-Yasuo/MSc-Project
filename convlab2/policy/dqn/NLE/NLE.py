# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import logging
import os
import json
from convlab2.nlg.template.multiwoz import TemplateNLG
from convlab2.policy.policy import Policy
from convlab2.policy.DQNModule import read_action_map
from convlab2.util.train_util import init_logging_handler
from convlab2.policy.vector.vector_multiwoz import MultiWozVector
from convlab2.util.file_util import cached_path
from convlab2.optimizer.radam import RAdam
import zipfile
import sys
import random
from transformers import Trainer, TrainingArguments, RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from nlp import Dataset

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(root_dir)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NLE(Policy):

    def __init__(self, domain='multiwoz'):
        self.domain = domain
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        if self.domain == 'multiwoz':
            check_point = os.path.join(root_dir, 'convlab2/policy/dqn/NLE/save/multiwoz/multiwoz_checkpoint')
        elif self.domain == 'taskmaster':
            check_point = os.path.join(root_dir, 'convlab2/policy/dqn/NLE/save/taskmaster/taskmaster_checkpoint')
        elif self.domain == 'script':
            check_point = os.path.join(root_dir, 'convlab2/policy/dqn/NLE/save/script/script_checkpoint')
        elif self.domain == 'personachat':
            check_point = os.path.join(root_dir, 'convlab2/policy/dqn/NLE/save/personachat/personachat_checkpoint')

        self.model = RobertaForSequenceClassification.from_pretrained(check_point).to(device=DEVICE)
        self.model.eval()
        self.nlg_usr = TemplateNLG(is_user=True)
        self.nlg_sys = TemplateNLG(is_user=False)
        self.get_score = nn.Softmax(dim=1)
        # load action mapping file
        action_map_file = os.path.join(root_dir, 'convlab2/policy/act_500_list.txt')
        _, self.ind2act_dict = read_action_map(action_map_file)
        # load vector for MultiWoz 2.1
        voc_file = os.path.join(root_dir, 'data/multiwoz/sys_da_voc.txt')
        voc_opp_file = os.path.join(root_dir, 'data/multiwoz/usr_da_voc.txt')
        self.vector = MultiWozVector(voc_file, voc_opp_file)

    def predict_ind(self, state):
        """Predict an system action and its index given state."""
        s_vec = torch.Tensor(self.vector.state_vectorize(state))
        user_action = state['user_action']
        if not user_action:
            action_pred = [['greet', 'general', 'none', 'none']]
            state['system_action'] = action_pred
            act_ind_pred = 494
            candidate_act_ind = [494]
            return action_pred, act_ind_pred, candidate_act_ind

        s_u = self.nlg_usr.generate(user_action)
        num_sys_act = len(self.ind2act_dict)
        s_u_list = [s_u] * num_sys_act
        s_a_list = []
        sys_action_list = []
        for a_ind in self.ind2act_dict:
            sys_action = self.vector.action_devectorize(self.ind2act_dict[a_ind])
            sys_action_list.append(sys_action)
            s_a = self.nlg_sys.generate(sys_action)
            s_a_list.append(s_a)
        encoding = self.tokenizer(s_u_list, s_a_list, padding=True, truncation=True, max_length=60)
        input_ids = torch.tensor(encoding['input_ids'])
        attention_mask = torch.tensor(encoding['attention_mask'])
        with torch.no_grad():
            logits = self.model(input_ids.to(device=DEVICE), attention_mask.to(device=DEVICE))[0].cpu().detach()

        relevance_score = self.get_score(logits)[:, 1].numpy()
        candidate_act_ind = np.argpartition(relevance_score, -5)[-5:]
        low_bound = relevance_score[np.argpartition(relevance_score, -5)[-5:]].min()
        upper_bound = relevance_score[np.argpartition(relevance_score, -5)[-5:]].max()
        if upper_bound - low_bound > 0.15:
            candidate_act_ind = np.argpartition(relevance_score, -3)[-3:]

        act_ind_pred = np.random.choice(candidate_act_ind)
        action_pred = sys_action_list[act_ind_pred]
        state['system_action'] = action_pred
        return action_pred, act_ind_pred, candidate_act_ind

    def predict(self, state):
        """Predict an system action and its index given state."""
        s_vec = torch.Tensor(self.vector.state_vectorize(state))
        user_action = state['user_action']
        if not user_action:
            action_pred = [['greet', 'general', 'none', 'none']]
            state['system_action'] = action_pred
            act_ind_pred = 494
            candidate_act_ind = [494]
            return action_pred, act_ind_pred, candidate_act_ind

        s_u = self.nlg_usr.generate(user_action)
        num_sys_act = len(self.ind2act_dict)
        s_u_list = [s_u] * num_sys_act
        s_a_list = []
        sys_action_list = []
        for a_ind in self.ind2act_dict:
            sys_action = self.vector.action_devectorize(self.ind2act_dict[a_ind])
            sys_action_list.append(sys_action)
            s_a = self.nlg_sys.generate(sys_action)
            s_a_list.append(s_a)
        encoding = self.tokenizer(s_u_list, s_a_list, padding=True, truncation=True, max_length=60)
        input_ids = torch.tensor(encoding['input_ids'])
        attention_mask = torch.tensor(encoding['attention_mask'])
        with torch.no_grad():
            logits = self.model(input_ids.to(device=DEVICE), attention_mask.to(device=DEVICE))[0].cpu().detach()

        relevance_score = self.get_score(logits)[:, 1].numpy()
        candidate_act_ind = np.argpartition(relevance_score, -5)[-5:]
        low_bound = relevance_score[np.argpartition(relevance_score, -5)[-5:]].min()
        upper_bound = relevance_score[np.argpartition(relevance_score, -5)[-5:]].max()
        if upper_bound - low_bound > 0.15:
            candidate_act_ind = np.argpartition(relevance_score, -3)[-3:]

        act_ind_pred = np.random.choice(candidate_act_ind)
        action_pred = sys_action_list[act_ind_pred]
        state['system_action'] = action_pred
        return action_pred

    def init_session(self):
        """Restore after one session"""
        pass


