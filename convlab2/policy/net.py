# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from convlab2.policy.rlmodule import Transition
from collections import namedtuple
import random

class duel_Q_network(nn.Module):
    def __init__(self, s_dim, h_dim, a_num):
        super(duel_Q_network, self).__init__()

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

    def ind2act(self, ind, mapping_action_file, da_dim):
        """
        :param ind:  int
        :param mapping_action_file: mapping file
        :param da_dim: int
        :return: [da_dim]
        """
        action = torch.zeros(da_dim)
        action[mapping_action_file[ind]] = 1
        return action

    def act2ind(self, act, mapping_action_file):
        """
        :param act:  [da_dim]
        :param mapping_action_file: mapping file
        :return: int
        """

        # np array: [da_dim,] => index
        positions = list(np.where(act == 1)[0])
        if positions in mapping_action_file:
            return mapping_action_file.index(positions)
        else:
            return False

    def select_action(self, s, epsilon, mapping_action_file, da_dim):
        """
        :param s: [s_dim]
        :return: [da_dim]
        """
        # index => action vector
        q_s_a = self.forward(s)
        if np.random.random_sample() > epsilon:
            a_ind = q_s_a.argmax().item()
        else:
            a_ind = np.random.choice(np.delete(np.arange(q_s_a.size(0)),q_s_a.argmax().item()))
        action = self.ind2act(a_ind, mapping_action_file, da_dim)
        return action


class ExperienceReplay(object):
    def __init__(self, max_size):
        # expert demonstration
        self.expert_demo = []
        # experience
        self.memory = []
        self.max_size = max_size

    def add_demo(self, *args):
        self.expert_demo.append(Transition(*args))

    def push(self, *args):
        """Saves a transition."""
        # add most recent experience to the start
        if self.__len__() < (self.max_size - len(self.expert_demo)):
            self.memory.insert(0, Transition(*args))
        else:
            # remove the earliest experience from the end
            self.memory.pop()
            self.memory.insert(0, Transition(*args))

    def get_expert_demo(self):
        return Transition(*zip(*self.expert_demo))

    def get_batch(self, batch_size=None):
        experience = self.expert_demo + self.memory
        if batch_size is None:
            return Transition(*zip(*experience))
        else:
            random_batch = random.sample(experience, batch_size)
            return Transition(*zip(*random_batch))

    def __len__(self):
        return len(self.memory)
