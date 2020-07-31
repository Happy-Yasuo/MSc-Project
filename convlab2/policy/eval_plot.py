# -*- coding: utf-8 -*-
import numpy as np
import plotly.graph_objects as go
import sys
import logging
import os
import datetime
import argparse
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_result/images')):
    os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_result/images'))


def load_eval_result(model_name):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_result/' + model_name + 'task_success.npy'), 'rb') as f:
        task_success = np.load(f)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_result/' + model_name + 'evaluator_success.npy'), 'rb') as f:
        evaluator_success = np.load(f)
    return task_success, evaluator_success


def plot_eval_result():
    model_config = {'DQN': {'print_name': 'DQN', 'line_config': dict(width=2, dash='solid')},
                    'DQfD_RE': {'print_name': 'Rule Expert', 'line_config': dict(width=2, dash='dash')},
                    'DQfD_NLE': {'print_name': 'No Label Expert', 'line_config': dict(width=2, dash='dashdot')}
                    }
    fig = go.Figure()
    for model_name in ['DQN', 'DQfD_RE', 'DQfD_NLE']:
        _, evaluator_success = load_eval_result(model_name)
        x0 = np.arange(0, 2.51, 0.01)
        y0 = evaluator_success[:251]
        fig.add_trace(go.Scatter(x=x0, y=y0, mode='lines',
                                 name=model_config[model_name]['print_name'],
                                 line=model_config[model_name]['line_config']))

    fig.update_xaxes(
        ticktext=["0.5M", "1.0M", "1.5M", "2.0M", "2.5M"],
        tickvals=["0.5", "1.0", "1.5", "2.0", "2.5"],
    )
    fig.update_yaxes(range=[0.0, 1.0])
    fig.update_xaxes(title_text='Steps')
    fig.update_yaxes(title_text='Avg Success')
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.4,
        xanchor="left",
        x=0.75
    ))
    fig.write_image(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_result/images/comparison.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="MultiWOZ", help="name of dataset")
    parser.add_argument("--model_name", type=str, default="PPO", help="name of model")
    parser.add_argument("--load_path", type=str, default='', help="path of model")
    parser.add_argument("--log_path_suffix", type=str, default="", help="suffix of path of log file")
    parser.add_argument("--log_dir_path", type=str, default="log", help="path of log directory")
    args = parser.parse_args()

    plot_eval_result()