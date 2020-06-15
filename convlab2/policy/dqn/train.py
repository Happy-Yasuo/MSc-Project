# -*- coding: utf-8 -*-
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
from torch import multiprocessing as mp
from convlab2.dialog_agent.agent import PipelineAgent
from convlab2.dialog_agent.env import Environment
from convlab2.dst.rule.multiwoz import RuleDST
from convlab2.policy.rule.multiwoz import RulePolicy
from convlab2.policy.dqn.DQfD import DQfD
from convlab2.policy.rlmodule import Transition
from convlab2.policy.net import ExperienceReplay
from convlab2.policy.vector.vector_multiwoz import MultiWozVector
from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator
from argparse import ArgumentParser

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def expert_demo_sample(env, policy, dialog_num=2000):
    voc_file = os.path.join(root_dir, 'data/multiwoz/sys_da_voc.txt')
    voc_opp_file = os.path.join(root_dir, 'data/multiwoz/usr_da_voc.txt')
    vector = MultiWozVector(voc_file, voc_opp_file)

    buff = ExperienceReplay(100000)
    sampled_num = 0
    sampled_traj_num = 0
    traj_len = 50
    real_traj_len = 0

    while sampled_traj_num < dialog_num:
        # for each trajectory, we reset the env and get initial state
        s = env.reset()
        for t in range(traj_len):
            # [s_dim] => [a_dim]
            s_vec = torch.Tensor(vector.state_vectorize(s))
            a = policy.predict(s)
            # interact with env
            next_s, r, done = env.step(a)
            # a flag indicates ending or not
            mask = 0 if done else 1
            next_s_vec = torch.Tensor(vector.state_vectorize(next_s))
            # save to queue
            buff.add_demo(s_vec.numpy(), vector.action_vectorize(a), r, next_s_vec.numpy(), mask)
            # update per step
            s = next_s
            real_traj_len = t
            if done:
                break

        # this is end of one trajectory
        sampled_num += real_traj_len
        sampled_traj_num += 1
        interval = dialog_num // 10
        if sampled_traj_num % interval == 0:
            print('The %d th expert demonstration generated.' % sampled_traj_num)
        # t indicates the valid trajectory length

    return buff


# simple rule DST
dst_sys = RuleDST()
expert_policy_sys = RulePolicy(character='sys')
# not use dst
dst_usr = None
# rule policy
policy_usr = RulePolicy(character='usr')
# assemble
simulator = PipelineAgent(None, None, policy_usr, None, 'user')
evaluator = MultiWozEvaluator()
env = Environment(None, simulator, None, dst_sys, evaluator)

# set dialogue number for demonstrations
pre_fill_buff = expert_demo_sample(env, expert_policy_sys, 2000)
# get expert data for compute auxiliary loss
expert_data = pre_fill_buff.get_expert_demo()
expert_s = list(expert_data.state)
expert_s_lst = [list(exp_s) for exp_s in expert_s]
expert_a = list(expert_data.action)


# interacts with env to get transitions
def sample(env, policy, pre_fill, num_frames=1000):
    buff = pre_fill
    sampled_num = 0
    sampled_traj_num = 0
    traj_len = 50
    real_traj_len = 0

    while sampled_num < num_frames:
        # for each trajectory, we reset the env and get initial state
        s = env.reset()
        for t in range(traj_len):
            # [s_dim] => [a_dim]
            s_vec = torch.Tensor(policy.vector.state_vectorize(s))
            a = policy.predict(s)
            # interact with env
            next_s, r, done = env.step(a)
            # a flag indicates ending or not
            mask = 0 if done else 1
            next_s_vec = torch.Tensor(policy.vector.state_vectorize(next_s))
            # save to queue
            buff.push(s_vec.numpy(), policy.vector.action_vectorize(a), r, next_s_vec.numpy(), mask)
            # update per step
            s = next_s
            real_traj_len = t
            if done:
                break

        # this is end of one trajectory
        sampled_num += real_traj_len
        sampled_traj_num += 1
        # print('%d frames sampled.' % sampled_num)
        # t indicates the valid trajectory length
    return buff, sampled_num


def update(policy, buff, batch_sz, expert_s_lst, expert_a):
    batch = buff.get_batch(batch_sz)
    s = torch.from_numpy(np.stack(batch.state)).to(device=DEVICE)
    a = torch.from_numpy(np.stack(batch.action)).to(device=DEVICE)
    r = torch.from_numpy(np.stack(batch.reward)).to(device=DEVICE)
    s_next = torch.from_numpy(np.stack(batch.next_state)).to(device=DEVICE)
    mask = torch.Tensor(np.stack(batch.mask)).to(device=DEVICE)
    policy.update(s.size(0), s, a, r, s_next, mask, expert_s_lst, expert_a)


policy_sys = DQfD(is_train=True)
# total number of frames
tot_frames = 2500000
train_batch_iter = 2000
batch_sz = 32
epoch = 1
sample_frames = 0
buff = pre_fill_buff
# pre-train
for _ in range(100):
    update(policy_sys, buff, batch_sz, expert_s_lst, expert_a)

while sample_frames < tot_frames:
    buff, new_frames = sample(env, policy_sys, buff, num_frames=1000)
    for iter in range(train_batch_iter):
        update(policy_sys, buff, batch_sz, expert_s_lst, expert_a)
    if epoch % 10 == 0:
        policy_sys.update_net()
        print('Epoch %d completed' % epoch)
        policy_sys.save('/cs/student/projects4/dsml/2019/jiaqwang/ConvLab-2/convlab2/policy/dqn/saved_model', epoch)
    sample_frames += new_frames
    epoch += 1





