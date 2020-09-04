# -*- coding: utf-8 -*-
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import numpy as np
import random
import torch
from torch import multiprocessing as mp
from convlab2.dialog_agent.agent import PipelineAgent
from convlab2.dialog_agent.env import Environment
from convlab2.dst.rule.multiwoz import RuleDST
from convlab2.policy.rule.multiwoz import RulePolicy
from convlab2.policy.dqn.NLE.DQfD import DQfD
from convlab2.policy.dqn.NLE.NLE import NLE
from convlab2.nlg.template.multiwoz import TemplateNLG
from convlab2.policy.DQNModule import read_action_map, Transition_NLE, ExperienceReplayNLE
from convlab2.policy.vector.vector_multiwoz import MultiWozVector
from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator
from transformers import Trainer, TrainingArguments, RobertaTokenizer, RobertaForSequenceClassification
from nlp import Dataset
from argparse import ArgumentParser
import logging


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sampler(env, policy, batchsz, expert):
    buff = ExperienceReplayNLE(100000)
    pos_action_buff = []
    neg_action_buff = []

    # we need to sample batchsz of (state, action, next_state, reward, mask)
    # each trajectory contains `trajectory_len` num of items, so we only need to sample
    # `batchsz//trajectory_len` num of trajectory totally
    # the final sampled number may be larger than batchsz.

    sampled_num = 0
    traj_len = 40  # max trajectory length

    while sampled_num < batchsz:
        # for each trajectory, we reset the env and get initial state
        s = env.reset()
        real_traj_len = 0   # real trajectory length of current sample dialog
        if expert:
            tmp_buff = ExperienceReplayNLE(100)
            tmp_action_buff = []
            tot_reward = 0
        for t in range(traj_len):
            # for expert policy
            s_vec = torch.Tensor(policy.vector.state_vectorize(s))
            if expert:
                a, a_ind, candidate_act_ind = policy.predict_ind(s)
            else:
                # [s_dim] => [a_dim]
                a, a_ind = policy.predict_ind(s)
            usr_a = s['user_action']
            sys_a = s['system_action']
            action_pair = (usr_a, sys_a)
            # interact with env
            next_s, r, done = env.step(a)

            # a flag indicates ending or not
            mask = 0 if done else 1
            next_s_vec = torch.Tensor(policy.vector.state_vectorize(next_s))

            if expert:
                # if expert action transformed to existing action space successfully, add this transition to expert demo
                tmp_buff.add_demo(s_vec.numpy(), a_ind, r, next_s_vec.numpy(), mask, 1, candidate_act_ind)
                tmp_action_buff.append(action_pair)
                tot_reward += r
            else:
                # add this transition to real experience memory
                buff.push(s_vec.numpy(), a_ind, r, next_s_vec.numpy(), mask, 0, [a_ind])
                real_traj_len += 1
            # update per step
            s = next_s
            # if dialog terminated then break
            if done:
                break
        if expert:
            buff.append(tmp_buff, True)
            real_traj_len += len(tmp_buff.expert_demo)
            if tot_reward >= 45:
                pos_action_buff += tmp_action_buff
            else:
                neg_action_buff += tmp_action_buff
        # this is end of one trajectory
        sampled_num += real_traj_len
    if len(neg_action_buff) > len(pos_action_buff):
        neg_action_buff = random.sample(neg_action_buff, len(pos_action_buff))
    return buff, pos_action_buff, neg_action_buff


def fine_tune(pos_action, neg_action, tokenizer, model):
    nlg_usr = TemplateNLG(is_user=True)
    nlg_sys = TemplateNLG(is_user=False)
    pos_train_usr_utter = []
    pos_train_sys_utter = []
    neg_train_usr_utter = []
    neg_train_sys_utter = []

    for turn in pos_action:
        if turn[0] != [] and turn[1] != []:
            s_u = nlg_usr.generate(turn[0])
            s_a = nlg_sys.generate(turn[1])
            pos_train_usr_utter.append(s_u)
            pos_train_sys_utter.append(s_a)
    for turn in neg_action:
        if turn[0] != [] and turn[1] != []:
            s_u = nlg_usr.generate(turn[0])
            s_a = nlg_sys.generate(turn[1])
            neg_train_usr_utter.append(s_u)
            neg_train_sys_utter.append(s_a)

    train_usr_utter = pos_train_usr_utter + neg_train_usr_utter
    train_sys_utter = pos_train_sys_utter + neg_train_sys_utter

    train_encoding = tokenizer(train_usr_utter, train_sys_utter, padding=True, truncation=True, max_length=80)
    train_encoding['label'] = [1] * len(pos_train_usr_utter) + [0] * len(neg_train_usr_utter)
    train_dataset = Dataset.from_dict(train_encoding)
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    save_dir = os.path.join(root_dir, 'convlab2/policy/dqn/NLE/save/script_fine_tune')
    log_dir = os.path.join(root_dir, 'convlab2/policy/dqn/NLE/save/script_fine_tune/logs')
    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=2,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=128,
        warmup_steps=500,
        weight_decay=0.01,
        evaluate_during_training=False,
        logging_dir=log_dir,
        )

    trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
    )
    trainer.train()
    trainer.save_model(os.path.join(save_dir, 'fine_tune_checkpoint'))


def pretrain(env, expert_policy, policy, batchsz, tokenizer):
    """
    pre-train agent policy
    :param env:
    :param expert_policy:
    :param policy:
    :param batchsz:
    :param process_num:
    :return:
    """
    # initialize pre-fill replay buffer
    prefill_buff = ExperienceReplayNLE(25000)
    pre_train_frames_num = 25000  # total number of dialogs required to sample
    seed = 20200721

    fine_tune_cnt = 0
    pos_fine_tune_buff = []
    neg_fine_tune_buff = []
    while len(prefill_buff.expert_demo) < pre_train_frames_num:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        seed += 1
        # achieve a buffer stored expert demonstrations
        new_buff, pos_action_pairs, neg_action_pairs = sampler(env, expert_policy, batchsz, True)
        pos_fine_tune_buff += pos_action_pairs
        neg_fine_tune_buff += neg_action_pairs
        cur_frames_num = len(list(new_buff.get_batch().mask))
        cur_success_num = list(new_buff.get_batch().reward).count(80)
        # put expert demonstrations to pre-fill buffer
        prefill_buff.append(new_buff, True)
        logging.debug('<<Replay Buffer>> At this turn, {} frames sampled with {} successful dialogues and {} fine tune '
                      'transitions, now pre-fill buffer has {} transitions in total'.format(cur_frames_num,
                        cur_success_num, len(pos_action_pairs), len(prefill_buff.expert_demo)))


        if fine_tune_cnt % 9 == 8:
            fine_tune(pos_fine_tune_buff, neg_fine_tune_buff, tokenizer, expert_policy.model)
            fine_tune_checkpoint = os.path.join(root_dir, 'convlab2/policy/dqn/NLE/save/script_fine_tune/fine_tune_checkpoint')
            expert_policy.model = RobertaForSequenceClassification.from_pretrained(fine_tune_checkpoint).to(device=DEVICE)
            logging.debug(
                '<<Fine Tune>> Epoch {} with {} successful transitions'.format(fine_tune_cnt, len(action_fine_tune_buff)))
        fine_tune_cnt += 1

    for epoch in range(25):
        pre_train_loss = 0
        # sample 3000 batches
        for _ in range(3000):
            # each batch size is 32
            batch = prefill_buff.get_batch(32)
            s = torch.from_numpy(np.stack(batch.state)).type(torch.float).to(device=DEVICE)
            a = torch.from_numpy(np.stack(batch.action)).type(torch.long).to(device=DEVICE)
            r = torch.from_numpy(np.stack(batch.reward)).type(torch.float).to(device=DEVICE)
            s_next = torch.from_numpy(np.stack(batch.next_state)).type(torch.float).to(device=DEVICE)
            mask = torch.Tensor(np.stack(batch.mask)).type(torch.float).to(device=DEVICE)
            expert_label = np.stack(batch.expert_label)
            candidate_a_ind = np.array(batch.candidate_act_ind)
            # compute loss for current batch
            cur_loss = policy.compute_loss(s, a, r, s_next, mask, expert_label, candidate_a_ind)
            pre_train_loss += cur_loss
            # update
            policy.update(cur_loss)
        # update target network
        policy.update_net()
        logging.debug('<<dialog policy DQfD pre-train>> Epoch {}, learning rate'
                      '{}, loss {}'.format(epoch, policy.scheduler.get_last_lr()[0], pre_train_loss/3000))
        # decay learning rate
        policy.scheduler.step()
    return prefill_buff


def train_update(prefill_buff, env, policy, batchsz, epoch):
    seed = epoch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # achieve a buffer stored real agent experience
    new_buff, _ = sampler(env, policy, batchsz, False)
    cur_frames_num = len(list(new_buff.get_batch().reward))
    cur_success_num = list(new_buff.get_batch().reward).count(80)
    # put real agent experience to pre-fill buffer while keep total transition number under maximum (100,000)
    prefill_buff.append(new_buff, False)
    train_loss = 0
    # linearly decrease epsilon
    if epoch < policy.epsilon_degrade_period:
        policy.epsilon = policy.epsilon_init - epoch * (policy.epsilon_init - policy.epsilon_final) / policy.epsilon_degrade_period
    else:
        policy.epsilon = policy.epsilon_final

    if (epoch+1) % 5 == 0:
        # update target network
        policy.update_net()

    # sample 3000 batches
    for _ in range(3000):
        # each batch size is 32
        batch = prefill_buff.get_batch(32)
        s = torch.from_numpy(np.stack(batch.state)).type(torch.float).to(device=DEVICE)
        a = torch.from_numpy(np.stack(batch.action)).type(torch.long).to(device=DEVICE)
        r = torch.from_numpy(np.stack(batch.reward)).type(torch.float).to(device=DEVICE)
        s_next = torch.from_numpy(np.stack(batch.next_state)).type(torch.float).to(device=DEVICE)
        mask = torch.Tensor(np.stack(batch.mask)).type(torch.float).to(device=DEVICE)
        expert_label = np.stack(batch.expert_label)
        candidate_a_ind = np.array(batch.candidate_act_ind)
        # compute loss for current batch
        cur_loss = policy.compute_loss(s, a, r, s_next, mask, expert_label, candidate_a_ind)
        train_loss += cur_loss
        # update
        policy.update(cur_loss)

    if epoch % 10 == 0:
        logging.debug('<<dialog policy DQfD train>> epoch {}, {} frames sampled with {} successful '
                      'dialogues at this turn, lr {}, loss: {}'.format(epoch, cur_frames_num, cur_success_num,
                                                                       policy.scheduler.get_last_lr()[0], train_loss/3000))
    # decay learning rate
    if policy.scheduler.get_last_lr()[0] > policy.min_lr:
        policy.scheduler.step()
    if epoch % 10 == 0:
        # save current model
        policy.save(os.path.join(root_dir, 'convlab2/policy/dqn/NLE/save/script_fine_tune'), epoch)


def generate_necessary_file(root_dir):
    voc_file = os.path.join(root_dir, 'data/multiwoz/sys_da_voc.txt')
    voc_opp_file = os.path.join(root_dir, 'data/multiwoz/usr_da_voc.txt')
    vector = MultiWozVector(voc_file, voc_opp_file)
    action_map_file = os.path.join(root_dir, 'convlab2/policy/act_500_list.txt')
    act2ind_dict, ind2act_dict = read_action_map(action_map_file)
    return vector, act2ind_dict, ind2act_dict


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--load_path", type=str, default="", help="path of model to load")
    parser.add_argument("--batchsz", type=int, default=1000, help="batch size of trajactory sampling")
    parser.add_argument("--epoch", type=int, default=2550, help="number of epochs to train")
    parser.add_argument("--process_num", type=int, default=1, help="number of processes of trajactory sampling")
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    check_point = os.path.join(root_dir, 'convlab2/policy/dqn/NLE/save/script/script_checkpoint')

    vector, act2ind_dict, ind2act_dict = generate_necessary_file(root_dir)
    # simple rule DST
    dst_usr = None
    dst_sys = RuleDST()
    # load policy sys
    policy_sys = DQfD(True)
    policy_sys.load(args.load_path)
    # expert
    expert_policy = NLE(domain='script')
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    # rule policy
    policy_usr = RulePolicy(character='usr')
    # assemble
    simulator = PipelineAgent(None, None, policy_usr, None, 'user')

    env = Environment(None, simulator, None, dst_sys)

    # pre-train
    prefill_buff = pretrain(env, expert_policy, policy_sys, args.batchsz, tokenizer)
    prefill_buff.max_size = 100000

    for i in range(args.epoch):
        train_update(prefill_buff, env, policy_sys, args.batchsz, i)
