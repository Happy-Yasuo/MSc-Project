# Imitation learning for task-oriented dialogue system
This work is based on ConvLab-2 framework.

## Algorithms
[DQfD](https://github.com/Happy-Yasuo/MSc-Project/tree/master/convlab2/policy/dqn)
This algorithm takes a rule-based expert's actions as demonstrations. 

## Action space
The state vector has 340 dimensions and action vector has 209 dimensions (each position in such a vector represent a single action). In order to introduce composite actions, 291 most common actions (including single and composite actions) in MultiWoz 2.1 dataset are mapped to the 209-dimension action space. Consequently, the total action number at each state is 500 (209 single actions and 291 composite actions) and we can map every action to the 209-dimension act space by the action mapping file.

Obviously, this 500 actions cannot cover all the actions might be taken by expert policy. Thus, if the expert policy takes an outside action, we will decompose it into multiple actions to ensure one of them can be found in our action space. The experiment has shown it does work. 


## Run the codes
Put files into corresponding path in ConvLab-2.

For Rule-based expert, run `train.py` in the `RE` directory:

```bash
cd ./convlab2/policy/dqn/RE
python train.py
```

For No-label expert, if NLE hasn't been trained before, run `train_NLE.py` in the `NLE` directory:
```bash
cd ./convlab2/policy/dqn/NLE
python train_NLE.py
```
Here is the evaluation result for the trained RoBERTa sequence classifier:
|  |    |
| -------------------|----------     |
|    eval_loss       |   0.2517561675531083      |
| eval_accuracy             |  0.9056347589952478    |
| eval_f1 | 0.9093754074846786       |
| eval_precision | 0.8747021196538317       |
| eval_recall | 0.9469110658520027       |

Then we can train the DQfD_NLE by running `train.py` in the same directory.
```bash
python train.py
```

For evaluating, run `evaluate.py` in the `policy` directory:

```bash
cd ./convlab2/policy
python evaluate.py --model_name DQfD_RE --load_path save/argument
python evaluate.py --model_name DQfD_NLE --load_path save/argument
```

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

Every 1,000 frames (steps), 2000 batches of size 32 would be sampled to train the model. I found it would be hard to optimize the loss if target network update period is 10,000 steps. Thus, now the update period is 5,000 steps. 



## Experiment Result
MLE is a supervised learning method which uses a simple feed forward network to fit the relationship between states and actions. It can be considered as a simple imitation learning and give PPO a warm start.

For MultiWoz 2.1 dataset, the optimum of MLE is 0.56 and PPO is 0.74. However, it seems there are some bugs in ConvLab-2 and the experiment result is shown as below. After fixing bugs and getting the optimum, the training trend of algorithms will be uploaded.

Now the 500 action space seems to be suitable for DQfD and after pretraining by 25,000 frames from expert demonstrations, it can get a task success rate of 0.84 with average reward of over 14. However, when the agent starts to interact with environment, both of the success rate and average reward decrease significantly and then increase slowly. Finally a success rate of 0.66 can be reached, which is still far from expectation. The experiments shows that the training result heavily depends on hyper-parameters (e.g. decay rate of learning rate scheduler and minimum learning rate), so I think this issue can be solved by parameter-tuning.   

|           | Task Success Rate |
| --------- | ----------------- |
| MLE       | 0.5157            |
| PPO       | 0.6136            |
| DQfD (epoch 0)      | 0.7600            |
| DQfD (epoch 1000)      | 0.8400           |
