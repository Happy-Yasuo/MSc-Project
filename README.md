# Imitation learning for task-oriented dialogue system
This work is based on ConvLab-2 framework.

## Experiment Result
MLE is a supervised learning method which uses a simple feed forward network to fit the relationship between states and actions. It can be considered as a simple imitation learning and give PPO a warm start.

For MultiWoz 2.1 dataset, the optimum of MLE is 0.56 and PPO is 0.74. However, it seems there are some bugs in ConvLab-2 and the experiment result is shown as below. After fixing bugs and getting the optimum, the training trend of algorithms will be uploaded.

|           | Task Success Rate |
| --------- | ----------------- |
| MLE       | 0.5157            |
| PPO       | 0.6136            |
| DQfD      | in training       |
