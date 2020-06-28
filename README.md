# Imitation learning for task-oriented dialogue system
This work is based on ConvLab-2 framework.

## Algorithms
[DQfD](https://github.com/Happy-Yasuo/MSc-Project/tree/master/convlab2/policy/dqn)

This algorithm takes a rule-based expert's actions as demonstrations. The state vector has 340 dimensions and action vector has 209 dimensions (each position in such a vector represent a single action). In order to introduce composite actions, 300 most common actions (including single and composite actions) in MultiWoz 2.1 dataset are mapped to the 209-dimension action space. Consequently, the total action number at each state is 300 and we can map every action to the 209-dimension act space by the action mapping file.


## Experimental Setup
As for DQfD, now a rule-based expert is used to generate demonstrations. The hyper-parameters mostly follows [Gordon-Hall et al.,
2020](https://arxiv.org/pdf/2004.08114.pdf) and shown as below:

|                    | Value         |
| -------------------|----------     |
| Steps              | 2,500,000     |
| Pre-training steps | 20,000        |
| Epsilon start      | 0.1           |
| Epsilon end        | 0.01          | 
| Epsilon decay rate | 500,000 steps |
| Discount factor    | 0.9           |
| Q network | 100d hidden layer and ReLU activation |
| Target network update period | 5,000 steps |
| Learning rate | 0.01 |
| L2 regularization weight  | 0.00001 |
| Max replay size | 100,000 |

Every 1,000 frames (steps), 2000 batches of size 32 would be sampled to train the model. I found it would be hard to optimize the loss if target network update period is 10,000 steps. Thus, now the update period is 5,000 steps. Since only 300 most common actions in MultiWoz 2.1 dataset is taken into consideration at each state, it is possible that some important actions are missing to complete the dialogue in some cases. For example, {booking slot: time, domain: hotel} and {booking slot: time, domain: attraction} are always hard to learn.

Network with more complex structure (two hidden layers) has been tested and it is not as good as network with 1 hidden layer(difficult to optimize loss.). 

Now action space of more action numbers is under training.

| Numbers of actions | Coverage in MultiWoz 2.1| Match rate for rule-based expert   |
| -------------------|-------------------------|----------------------------------- |
| 300                | 63.74%                  | 66.23%                             |
| 400                | 67.84%                  | 73.19%                             |
| 500                | 70.87%                  | 79.28%                             |
| 600                | 73.26%                  | 79.84%                             |



## Experiment Result
MLE is a supervised learning method which uses a simple feed forward network to fit the relationship between states and actions. It can be considered as a simple imitation learning and give PPO a warm start.

For MultiWoz 2.1 dataset, the optimum of MLE is 0.56 and PPO is 0.74. However, it seems there are some bugs in ConvLab-2 and the experiment result is shown as below. After fixing bugs and getting the optimum, the training trend of algorithms will be uploaded.

|           | Task Success Rate |
| --------- | ----------------- |
| MLE       | 0.5157            |
| PPO       | 0.6136            |
| DQfD      | in training       |
