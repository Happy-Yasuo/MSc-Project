# -*- coding: utf-8 -*-
import numpy as np
import torch
import random
from torch import multiprocessing as mp
from convlab2.dialog_agent.agent import PipelineAgent
from convlab2.dialog_agent.session import BiSession
from convlab2.dialog_agent.env import Environment
from convlab2.dst.rule.multiwoz import RuleDST
from convlab2.policy.rule.multiwoz import RulePolicy
from convlab2.policy.rlmodule import Memory, Transition
from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator
from pprint import pprint
import json
import matplotlib.pyplot as plt
import sys
import logging
import os
import datetime
import argparse

def init_logging(log_dir_path, path_suffix=None):
    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    if path_suffix:
        log_file_path = os.path.join(log_dir_path, f"{current_time}_{path_suffix}.log")
    else:
        log_file_path = os.path.join(log_dir_path, "{}.log".format(current_time))

    stderr_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file_path)
    format_str = "%(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s"
    logging.basicConfig(level=logging.DEBUG, handlers=[stderr_handler, file_handler], format=format_str)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(dataset_name, model_name, load_path):
    seed = 20200722
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if dataset_name == 'MultiWOZ':
        dst_sys = RuleDST()
        
        if model_name == "PPO":
            from convlab2.policy.ppo import PPO
            if load_path:
                policy_sys = PPO(False)
                policy_sys.load(load_path)
            else:
                policy_sys = PPO.from_pretrained()
        elif model_name == "DQN":
            from convlab2.policy.dqn.DQN.DQN import DQN
            if load_path:
                policy_sys = DQN(False)
                policy_sys.load(load_path)
            else:
                print('Please add load path.')
        elif model_name == "DQfD_RE":
            from convlab2.policy.dqn.RE.DQfD import DQfD
            if load_path:
                policy_sys = DQfD(False)
                policy_sys.load(load_path)
            else:
                print('Please add load path.')
        elif model_name == "DQfD_NLE":
            from convlab2.policy.dqn.NLE.DQfD import DQfD
            if load_path:
                policy_sys = DQfD(False)
                policy_sys.load(load_path)
            else:
                print('Please add load path.')
        elif model_name == "MLE":
            from convlab2.policy.mle.multiwoz import MLE
            if load_path:
                policy_sys = MLE()
                policy_sys.load(load_path)
            else:
                policy_sys = MLE.from_pretrained()


        policy_usr = RulePolicy(character='usr')
        simulator = PipelineAgent(None, None, policy_usr, None, 'user')


        agent_sys = PipelineAgent(None, dst_sys, policy_sys, None, 'sys')

        evaluator = MultiWozEvaluator()
        sess = BiSession(agent_sys, simulator, None, evaluator)

        task_success = 0
        evaluator_success = 0
        for seed in range(100):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            sess.init_session()
            sys_response = []

            cur_success = 0
            for i in range(40):
                sys_response, user_response, session_over, reward = sess.next_turn(sys_response)
                if reward == 80:
                    cur_success = 1
                    task_success += 1
                if session_over is True:
                    break
            # logging.debug('Current task success: {}, the evaluator result: {}.'.format(cur_success, sess.evaluator.task_success()))
            evaluator_success += sess.evaluator.task_success()

        logging.debug('Task success rate: {} and evaluator result: {}.'.format(task_success/100, evaluator_success/100))
        return task_success/100, evaluator_success/100


root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def evaluate_trend(dataset_name, model_name):
    task_success_list = []
    evaluator_success_list = []
    for epoch in range(0, 2010, 10):
        load_path = "save/" + str(epoch)
        task_success_rate, evaluator_success_rate = evaluate(dataset_name, model_name, load_path)
        task_success_list.append(task_success_rate)
        evaluator_success_list.append(evaluator_success_rate)
    array_1 = np.array(task_success_list)
    array_2 = np.array(evaluator_success_list)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_result/' + model_name + 'task_success.npy'), 'wb') as f:
        np.save(f, array_1)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_result/' + model_name + 'evaluator_success.npy'), 'wb') as f:
        np.save(f, array_2)
    logging.debug('{} epochs completed.'.format(epoch))


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="MultiWOZ", help="name of dataset")
    parser.add_argument("--model_name", type=str, default="PPO", help="name of model")
    parser.add_argument("--load_path", type=str, default='', help="path of model")
    parser.add_argument("--log_path_suffix", type=str, default="", help="suffix of path of log file")
    parser.add_argument("--log_dir_path", type=str, default="log", help="path of log directory")
    args = parser.parse_args()

    init_logging(log_dir_path=args.log_dir_path, path_suffix=args.log_path_suffix)
    evaluate_trend(
        dataset_name=args.dataset_name,
        model_name=args.model_name
    )