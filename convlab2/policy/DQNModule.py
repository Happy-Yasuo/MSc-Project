# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple
import random
import json
import copy


class DuelDQN(nn.Module):
    def __init__(self, s_dim, h_dim, a_num):
        super(DuelDQN, self).__init__()

        self.net = nn.Sequential(nn.Linear(s_dim, h_dim),
                                 nn.ReLU(),)
        self.v_stream = nn.Linear(h_dim, 1)
        self.adv_stream = nn.Linear(h_dim, a_num)

    def forward(self, s):
        h = self.net(s)
        v = self.v_stream(h)
        adv = self.adv_stream(h)
        q = v + adv-adv.mean()
        return q

    def ind2act(self, ind, act_vec_dict):
        """
        :param act_vec_dict:
        :param ind:  int
        :return: [da_dim]
        """
        action = random.choice(act_vec_dict[str(ind)])
        return action

    def select_action(self, s, epsilon, act_vec_dict):
        """
        :param s: [s_dim]
        :param epsilon:
        :param act_vec_dict:
        :return: [da_dim], [1]
        """
        q_s_a = self.forward(s)
        if np.random.random_sample() > epsilon:
            a_ind = q_s_a.argmax().item()
        else:
            a_ind = np.random.choice(np.delete(np.arange(q_s_a.size(0)), q_s_a.argmax().item()))
        action = self.ind2act(a_ind, act_vec_dict)
        return action, a_ind

    def select_action_ind(self, s, epsilon):
        """
        :param s: [s_dim]
        :param epsilon:
        :return: [1]
        """
        q_s_a = self.forward(s)
        if np.random.random_sample() > epsilon:
            a_ind = q_s_a.argmax().item()
        else:
            a_ind = np.random.choice(np.delete(np.arange(q_s_a.size(0)), q_s_a.argmax().item()))
        return a_ind


def read_action_map(file_path, da_dim=209):
    with open(file_path, 'r') as f:
        position_dict = json.load(f)
    act_vec_dict = copy.deepcopy(position_dict)
    for i in act_vec_dict:
        for ind, j in enumerate(act_vec_dict[i]):
            cur_act_vec = np.zeros(da_dim)
            cur_act_vec[j] = 1
            act_vec_dict[i][ind] = cur_act_vec
    return position_dict, act_vec_dict


def expert_act_vec2ind(cur_exp_act, position_dict, rand='True'):
    """transform an expert action into an index in DQN action space"""
    """cur_exp_act is the indexes list of an expert action vector with 209 dimensions"""
    num_act = len(cur_exp_act)
    # -1 means we fail to map the expert action into an index in our action space
    cur_act_ind = -1
    num_same_action = {}
    if num_act == 1:
        cur_act_ind = list(position_dict.keys())[list(position_dict.values()).index([cur_exp_act])]
    elif num_act > 1:
        cur_exp_set = set(cur_exp_act)
        for idx in position_dict:
            cur_comb_len = len(position_dict[idx][0])
            if cur_comb_len > num_act:
                continue
            elif cur_comb_len == num_act:
                for act_vec in position_dict[idx]:
                    if act_vec == cur_exp_act:
                        cur_act_ind = idx
                        break
                if cur_act_ind != -1:
                    break
            else:
                best_similarity = 0
                for act_vec in position_dict[idx]:
                    act_vec_set = set(act_vec)
                    if act_vec_set <= cur_exp_set:
                        similarity = len(act_vec_set & cur_exp_set)
                        if similarity > best_similarity:
                            best_similarity = similarity
                    else:
                        continue
                num_same_action[idx] = best_similarity

    if num_act > 0:
        if cur_act_ind == -1:
            similarity_lst = list(num_same_action.items())
            similarity_lst.sort(key=lambda x: x[1])
            similarity_lst.reverse()
            candidate = []
            max_score = similarity_lst[0][1]
            for i in range(list(num_same_action.values()).count(max_score)):
                candidate.append(similarity_lst[i][0])
            if max_score > num_act-2:
                if rand:
                    cur_act_ind = random.choice(candidate)
                else:
                    cur_act_ind = candidate[0]
            else:
                cur_act_ind = -1
    else:
        cur_act_ind = -1
    return int(cur_act_ind)


# expert_label is used to record if a transition is from expert demonstration
Transition_new = namedtuple('Transition_new', ('state', 'action', 'reward', 'next_state', 'mask', 'expert_label'))


class ExperienceReplay(object):
    def __init__(self, max_size):
        # expert demonstration
        self.expert_demo = []
        # experience
        self.memory = []
        self.max_size = max_size

    def add_demo(self, *args):
        """use this method to add expert demonstrations """
        self.expert_demo.append(Transition_new(*args))

    def push(self, *args):
        """use this method to add real experience"""
        self.memory.append(Transition_new(*args))

    def append(self, new_memory):
        """use this method to add new memory from interacting with environment and keep the total size under maximum"""
        self.memory = new_memory.memory + self.memory
        if self.__len__() > (self.max_size - len(self.expert_demo)):
            num_del = self.__len__() - (self.max_size - len(self.expert_demo))
            for _ in range(num_del):
                self.memory.pop()

    def get_batch(self, batch_size=None):
        all_data = self.expert_demo + self.memory
        if batch_size is None:
            return Transition_new(*zip(*all_data))
        else:
            random_batch = random.sample(all_data, batch_size)
            return Transition_new(*zip(*random_batch))

    def __len__(self):
        # current length of experience
        return len(self.memory)
