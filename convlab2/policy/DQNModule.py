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

    def ind2act(self, ind, ind2act_dict):
        """
        :param ind2act_dict:
        :param ind:  int
        :return: [da_dim]
        """
        action = ind2act_dict[ind]
        return action

    def select_action(self, s, epsilon, ind2act_dict):
        """
        :param s: [s_dim]
        :param epsilon:
        :param ind2act_dict:
        :return: [da_dim], [1]
        """
        q_s_a = self.forward(s)
        if np.random.random_sample() > epsilon:
            a_ind = q_s_a.argmax().item()
        else:
            a_ind = np.random.choice(np.delete(np.arange(q_s_a.size(0)), q_s_a.argmax().item()))
        action = self.ind2act(a_ind, ind2act_dict)
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
        act_lst = f.readlines()
        for i in range(len(act_lst)):
            act_lst[i] = act_lst[i].strip('\n').split('-')
    for i in range(len(act_lst)):
        if act_lst[i] != ['']:
            act_lst[i] = tuple(map(int, act_lst[i]))
        else:
            act_lst[i] = ()
    act2ind_dict = dict(zip(act_lst, list(range(len(act_lst)))))
    ind2act_dict = {}
    for i in range(len(act_lst)):
        act_vec = np.zeros(da_dim)
        act_vec[list(act_lst[i])] = 1
        ind2act_dict[i] = act_vec
    return act2ind_dict, ind2act_dict


def expert_act_vec2ind(cur_exp_act, act2ind_dict):
    """transform an expert action into an index in DQN action space"""
    """cur_exp_act is the indexes tuple of an expert action vector with 209 dimensions"""
    # initialize cur_act_ind
    # -1 means we fail to map the expert action into an index in our action space
    cur_act_ind = -1
    if cur_exp_act in act2ind_dict.keys():
        cur_act_ind = act2ind_dict[cur_exp_act]
    return cur_act_ind


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

    def append(self, new_memory, expert=False):
        """use this method to add new memory from interacting with environment and keep the total size under maximum"""
        if expert:
            self.expert_demo = new_memory.expert_demo + self.expert_demo
        else:
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
