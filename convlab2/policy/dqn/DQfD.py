# -*- coding: utf-8 -*-
import torch
from torch import optim
import numpy as np
import logging
import os
import json
from convlab2.policy.policy import Policy
from convlab2.policy.net import duel_Q_network
from convlab2.util.train_util import init_logging_handler
from convlab2.policy.vector.vector_multiwoz import MultiWozVector
from convlab2.util.file_util import cached_path
import zipfile
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQfD(Policy):

    def __init__(self, is_train=False, dataset='Multiwoz'):

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json'), 'r') as f:
            cfg = json.load(f)
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mapping_action.json'), 'r') as f2:
            self.mapping_action_file = list(json.load(f2).values())
        self.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg['save_dir'])
        self.gamma = cfg['gamma']
        self.epsilon = cfg['epsilon']
        self.tau = cfg['tau']
        self.is_train = is_train
        if is_train:
            init_logging_handler(os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg['log_dir']))

        if dataset == 'Multiwoz':
            voc_file = os.path.join(root_dir, 'data/multiwoz/sys_da_voc.txt')
            voc_opp_file = os.path.join(root_dir, 'data/multiwoz/usr_da_voc.txt')
            self.vector = MultiWozVector(voc_file, voc_opp_file)
            # total number of actions considered
            self.action_number = cfg['action_number']
            # current Q network to be trained
            self.Q = duel_Q_network(self.vector.state_dim, cfg['h_dim'], self.action_number).to(device=DEVICE)
            # target Q network
            self.target_Q = duel_Q_network(self.vector.state_dim, cfg['h_dim'], self.action_number).to(device=DEVICE)

        if is_train:
            self.optimizer = optim.Adam(self.Q.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
            self.criterion = torch.nn.MSELoss()

    def predict(self, state):
        """
        Predict an system action given state.
        Args:
            state (dict): Dialog state. Please refer to util/state.py
        Returns:
            action : System act, with the form of (act_type, {slot_name_1: value_1, slot_name_2, value_2, ...})
        """
        s_vec = torch.Tensor(self.vector.state_vectorize(state))
        a = self.Q.select_action(s_vec.to(device=DEVICE), self.epsilon, self.mapping_action_file, self.vector.da_dim).cpu()
        action = self.vector.action_devectorize(a.numpy())
        state['system_action'] = action
        return action

    def init_session(self):
        """
        Restore after one session
        """
        pass

    def act_diff(self, batch_a_e):
        """
        :param batch_a_e: a batch of actions taken by expert
        :return: [batch_len, tot_action_num]
        """
        act_diff = torch.zeros([len(batch_a_e), self.action_number])
        act_diff[list(range(len(batch_a_e))), batch_a_e] = self.tau
        return act_diff

    def aux_loss(self, s, expert_s_lst, expert_a):
        """
        s: state tensor
        expert_s_lst: list of state of expert
        expert_a: list of actions taken by expert
        """

        batch_s = []
        batch_a_e = []
        batch_a_e_gather = []
        s_array = s.cpu().numpy()
        for batch_id in range(s.size(0)):
            s_lst = list(s_array[batch_id])
            # only keep those states can be found in expert demonstration
            if s_lst in expert_s_lst:
                act_expert_index = self.Q.act2ind(expert_a[expert_s_lst.index(s_lst)], self.mapping_action_file)
                batch_s.append(s_array[batch_id])
                batch_a_e.append(act_expert_index)
                batch_a_e_gather.append([act_expert_index])
        new_s = torch.from_numpy(np.stack(batch_s)).to(device=DEVICE)
        new_q_preds = self.Q.forward(new_s)
        diff = self.act_diff(batch_a_e).to(device=DEVICE)
        max_act_q_preds = torch.max((new_q_preds + diff), dim=1)[0]
        expert_act_q_preds = new_q_preds.gather(-1, torch.tensor(batch_a_e_gather).to(device=DEVICE)).squeeze(-1)
        aux_loss = (max_act_q_preds - expert_act_q_preds).sum() / s.size(0)
        return aux_loss

    def update_net(self):
        self.target_Q.load_state_dict(self.Q.state_dict())

    def update(self, batchsz, s, a, r, s_next, mask, expert_s_lst, expert_a):

        batch_act = a.cpu().numpy()
        act_index = []
        retain_index = []
        for i in range(batchsz):
            # action: vector form => index in Q value vector
            # only keep actions in action space of mapping action file and
            # corresponding state, reward, next state and mask
            cur_act_ind = self.Q.act2ind(batch_act[i], self.mapping_action_file)
            if cur_act_ind:
                retain_index.append(i)
                act_index.append([cur_act_ind])
        action_idx = torch.tensor(act_index).to(device=DEVICE)

        q_preds = self.Q.forward(s[retain_index])
        with torch.no_grad():
            # online net for action selection in next state
            online_next_q_preds = self.Q.forward(s_next[retain_index])
            # target net for q value predicting in the next state
            next_q_preds = self.target_Q.forward(s_next[retain_index])
        act_q_preds = q_preds.gather(-1, action_idx).squeeze(-1)
        # use online net to choose action for the next state
        online_actions = online_next_q_preds.argmax(dim=-1, keepdim=True)
        # use target net to predict the corresponding value
        max_next_q_preds = next_q_preds.gather(-1, online_actions).squeeze(-1)
        max_q_targets = r[retain_index] + self.gamma * max_next_q_preds * mask[retain_index]
        q_loss = self.criterion(act_q_preds, max_q_targets)
        try:
            aux_loss_term = self.aux_loss(s[retain_index], expert_s_lst, expert_a)  # get expert demo
            loss = q_loss + aux_loss_term
        except:
            print('aux loss not applicable')
            loss = q_loss
        print('Training loss: %.3f' %loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, directory, epoch):
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.Q.state_dict(), directory + '/' + str(epoch) + '_dqn.mdl')

        logging.info('<<dialog policy>> epoch {}: saved network to mdl'.format(epoch))
    
    def load(self, filename):
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

