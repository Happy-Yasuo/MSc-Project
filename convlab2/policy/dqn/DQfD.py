# -*- coding: utf-8 -*-
import torch
from torch import optim
import numpy as np
import logging
import os
import json
from convlab2.policy.policy import Policy
from convlab2.policy.DQNModule import DuelDQN, read_action_map
from convlab2.util.train_util import init_logging_handler
from convlab2.policy.vector.vector_multiwoz import MultiWozVector
from convlab2.util.file_util import cached_path
import zipfile
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQfD(Policy):

    def __init__(self, dataset='Multiwoz'):
        # load configuration file
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json'), 'r') as f:
            cfg = json.load(f)
        self.gamma = cfg['gamma']
        self.epsilon_init = cfg['epsilon_init']
        self.epsilon_final = cfg['epsilon_final']
        self.epsilon = self.epsilon_init
        self.epsilon_degrade_period = cfg['epsilon_degrade_period']
        self.tau = cfg['tau']
        self.action_number = cfg['action_number']  # total number of actions considered
        init_logging_handler(os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg['log_dir']))
        # load action mapping file
        action_map_file = os.path.join(root_dir, 'convlab2/policy/act_list.txt')
        _, self.ind2act_dict = read_action_map(action_map_file)
        # load vector for MultiWoz 2.1
        if dataset == 'Multiwoz':
            voc_file = os.path.join(root_dir, 'data/multiwoz/sys_da_voc.txt')
            voc_opp_file = os.path.join(root_dir, 'data/multiwoz/usr_da_voc.txt')
            self.vector = MultiWozVector(voc_file, voc_opp_file)
        # build Q network
        # current Q network to be trained
        self.Q = DuelDQN(self.vector.state_dim, cfg['h_dim'], self.action_number).to(device=DEVICE)
        # target Q network
        self.target_Q = DuelDQN(self.vector.state_dim, cfg['h_dim'], self.action_number).to(device=DEVICE)
        # define optimizer
        self.optimizer = optim.Adam(self.Q.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        # loss function
        self.criterion = torch.nn.MSELoss()

    def predict(self, state):
        """Predict an system action and its index given state."""
        s_vec = torch.Tensor(self.vector.state_vectorize(state))
        a, a_ind = self.Q.select_action(s_vec.to(device=DEVICE), self.epsilon, self.ind2act_dict)
        action = self.vector.action_devectorize(a)
        state['system_action'] = action
        return action, a_ind

    def predict_ind(self, state):
        """Predict an action index action space given state."""
        s_vec = torch.Tensor(self.vector.state_vectorize(state))
        a_ind = self.Q.select_action_ind(s_vec.to(device=DEVICE), self.epsilon, self.ind2act_dict)
        return a_ind

    def init_session(self):
        """Restore after one session"""
        pass

    def aux_loss(self, s, a, expert_label):
        """compute auxiliary loss given batch of states, actions and expert labels"""
        # only keep those expert demonstrations by setting expert label to 1
        s_exp = s[np.where(expert_label == 1)[0]]
        a_exp = a[np.where(expert_label == 1)[0]]
        # if there exist expert demonstration in current batch
        if s_exp.size(0) > 0:
            # compute q value predictions for states for each action
            q_all = self.Q(s_exp)
            # only when agent take the same action as the expert does, the act_diff term(i.e. l(a_e,a)) is 0
            act_diff = q_all.new_full(q_all.size(), self.tau)
            act_diff[list(range(q_all.size(0))), a_exp] = 0
            # aux_loss = max(Q(s, a) + l(a_e, a)) - Q(s, a_e)
            q_max_act = (q_all + act_diff).max(dim=1)[0]
            q_exp_act = q_all.gather(-1, a_exp.unsqueeze(1)).squeeze(-1)
            aux_loss = (q_max_act - q_exp_act).sum() / s_exp.size(0)
        else:
            aux_loss = 0
        return aux_loss

    def update_net(self):
        """update target network by copying parameters from online network"""
        self.target_Q.load_state_dict(self.Q.state_dict())

    def compute_loss(self, s, a, r, s_next, mask, expert_label):
        """compute loss for batch"""
        # q value predictions for current state for each action
        q_preds = self.Q(s)
        with torch.no_grad():
            # online net for action selection in next state
            online_next_q_preds = self.Q(s_next)
            # target net for q value predicting in the next state
            next_q_preds = self.target_Q(s_next)
        # select q value predictions for corresponding actions
        act_q_preds = q_preds.gather(-1, a.unsqueeze(1)).squeeze(-1)
        # use online net to choose action for the next state
        online_actions = online_next_q_preds.argmax(dim=-1, keepdim=True)
        # use target net to predict the corresponding value
        max_next_q_preds = next_q_preds.gather(-1, online_actions).squeeze(-1)
        # compute target q values
        max_q_targets = r + self.gamma * max_next_q_preds * mask
        # q loss
        q_loss = self.criterion(act_q_preds, max_q_targets)
        # auxiliary loss
        aux_loss_term = self.aux_loss(s, a, expert_label)
        # total loss
        loss = q_loss + aux_loss_term
        return loss

    def update(self, loss):
        """update online network"""
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, directory, epoch):
        """save model to directory"""
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.Q.state_dict(), directory + '/' + str(epoch) + '_dqn.mdl')

        logging.info('<<dialog policy>> epoch {}: saved network to mdl'.format(epoch))
    
    def load(self, filename):
        """load model"""
        dqn_mdl_candidates = [
            filename + '_dqn.mdl',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '_dqn.mdl')
        ]
        for dqn_mdl in dqn_mdl_candidates:
            if os.path.exists(dqn_mdl):
                self.Q.load_state_dict(torch.load(dqn_mdl, map_location=DEVICE))
                self.target_Q.load_state_dict(torch.load(dqn_mdl, map_location=DEVICE))
                logging.info('<<dialog policy>> loaded checkpoint from file: {}'.format(dqn_mdl))
                break
