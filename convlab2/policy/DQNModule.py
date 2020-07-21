# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple
import random
import json
import copy
import logging


class DuelDQN(nn.Module):
    def __init__(self, s_dim, h_dim, a_num):
        super(DuelDQN, self).__init__()

        self.net = nn.Linear(s_dim, h_dim)
        self.activation = nn.ReLU()
        self.v_stream = nn.Linear(h_dim, 1)
        self.adv_stream = nn.Linear(h_dim, a_num)
        self.init_weights()

    def init_weights(self):
        self.net.weight.data.normal_(0.0, 0.01)
        self.net.bias.data.fill_(0.0)
        self.v_stream.weight.data.normal_(0.0, 0.01)
        self.v_stream.bias.data.fill_(0.0)
        self.adv_stream.weight.data.normal_(0.0, 0.01)
        self.adv_stream.bias.data.fill_(0.0)

    def forward(self, s):
        h = self.activation(self.net(s))
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

    def select_action(self, s, epsilon, ind2act_dict, istrain):
        """

        :param s: [s_dim]
        :param epsilon:
        :param ind2act_dict:
        :return: [da_dim], [1]
        :param istrain: Bool
        """
        q_s_a = self.forward(s).cpu().detach()
        if istrain:
            if np.random.random_sample() > epsilon:
                a_ind = q_s_a.argmax().item()
            else:
                a_ind = np.random.choice(np.delete(np.arange(q_s_a.size(-1)), q_s_a.argmax().item()))
        else:
            a_ind = q_s_a.argmax().item()
        action = self.ind2act(a_ind, ind2act_dict)
        return action, a_ind


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
    num_act = len(cur_exp_act)
    # -1 means we fail to map the expert action into an index in our action space
    cur_act_ind = -1
    num_same_action = {}
    retain_pos = []
    if num_act == 1:
        cur_act_ind = list(act2ind_dict.values())[list(act2ind_dict.keys()).index(cur_exp_act)]
        retain_pos = [0]
    elif num_act > 1:
        cur_exp_set = set(cur_exp_act)
        for idx in act2ind_dict:
            cur_comb_len = len(idx)
            if cur_comb_len > num_act:
                continue
            elif cur_comb_len == num_act:
                if idx == cur_exp_act:
                    cur_act_ind = act2ind_dict[idx]
                    retain_pos = list(range(num_act))
                    break
            else:
                similarity = 0
                act_vec_set = set(idx)
                if act_vec_set <= cur_exp_set:
                    similarity = len(act_vec_set & cur_exp_set)
                else:
                    continue
                num_same_action[idx] = similarity
    else:
        cur_act_ind = -1

    if num_act > 0:
        if cur_act_ind == -1:
            best_match_act = max(num_same_action, key=num_same_action.get)
            cur_act_ind = act2ind_dict[best_match_act]
            for single_act in best_match_act:
                for pos in range(len(cur_exp_act)):
                    if cur_exp_act[pos] == single_act:
                        retain_pos.append(pos)

    return int(cur_act_ind), retain_pos


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
        self.expert_demo.insert(0, Transition_new(*args))

    def push(self, *args):
        """use this method to add real experience"""
        self.memory.insert(0, Transition_new(*args))

    def append(self, new_memory, expert=False):
        """use this method to add new memory from interacting with environment and keep the total size under maximum"""
        if expert:
            self.expert_demo = new_memory.expert_demo + self.expert_demo
            if len(self.expert_demo) > self.max_size:
                num_del = len(self.expert_demo) - self.max_size
                logging.debug('<<Replay Buffer>> {} expert transitions newly appended and {} expert '
                              'transitions deleted,'.format(len(new_memory.expert_demo), num_del))
                for _ in range(num_del):
                    self.expert_demo.pop()
        else:
            self.memory = new_memory.memory + self.memory
            if self.__len__() > (self.max_size - len(self.expert_demo)):
                num_del = self.__len__() - (self.max_size - len(self.expert_demo))
                logging.debug('<<Replay Buffer>> {} transitions newly appended and {} transitions deleted,'.format(
                    len(new_memory.memory), num_del))
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


Transition_NLE = namedtuple('Transition_NLE', ('state', 'action', 'reward', 'next_state', 'mask', 'expert_label', 'candidate_act_ind'))


class ExperienceReplayNLE(object):
    def __init__(self, max_size):
        # expert demonstration
        self.expert_demo = []
        # experience
        self.memory = []
        self.max_size = max_size

    def add_demo(self, *args):
        """use this method to add expert demonstrations """
        self.expert_demo.insert(0, Transition_NLE(*args))

    def push(self, *args):
        """use this method to add real experience"""
        self.memory.insert(0, Transition_NLE(*args))

    def append(self, new_memory, expert=False):
        """use this method to add new memory from interacting with environment and keep the total size under maximum"""
        if expert:
            self.expert_demo = new_memory.expert_demo + self.expert_demo
            if len(self.expert_demo) > self.max_size:
                num_del = len(self.expert_demo) - self.max_size
                logging.debug('<<Replay Buffer>> {} expert transitions newly appended and {} expert '
                              'transitions deleted,'.format(len(new_memory.expert_demo), num_del))
                for _ in range(num_del):
                    self.expert_demo.pop()
        else:
            self.memory = new_memory.memory + self.memory
            if self.__len__() > (self.max_size - len(self.expert_demo)):
                num_del = self.__len__() - (self.max_size - len(self.expert_demo))
                logging.debug('<<Replay Buffer>> {} transitions newly appended and {} transitions deleted,'.format(
                    len(new_memory.memory), num_del))
                for _ in range(num_del):
                    self.memory.pop()

    def get_batch(self, batch_size=None):
        all_data = self.expert_demo + self.memory
        if batch_size is None:
            return Transition_NLE(*zip(*all_data))
        else:
            random_batch = random.sample(all_data, batch_size)
            return Transition_NLE(*zip(*random_batch))

    def __len__(self):
        # current length of experience
        return len(self.memory)
