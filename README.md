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

For DQN, run `train.py` in the `DQN` directory:

```bash
cd ./convlab2/policy/dqn/DQN
python train.py
```

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

For evaluating the task success rate and other algorithm performance, run `evaluate.py` in the `policy` directory:

```bash
cd ./convlab2/policy
python evaluate.py --model_name DQN --load_path save/argument
python evaluate.py --model_name DQfD_RE --load_path save/argument
python evaluate.py --model_name DQfD_NLE --load_path save/argument
```

For reproduce a figure as below, we need firstly complete the training of corresponding agents and then run
```bash
cd ./convlab2/policy
python evaluate_success.py --model_name DQN
python evaluate_success.py --model_name DQfD_RE
python evaluate_success.py --model_name DQfD_NLE
```

Then, run
```bash
cd ./convlab2/policy
python eval_plot.py
```
to generate the figure plot in '/eval_result/images' directory.

![image](https://github.com/JQWang-77/MSc-Project/blob/master/convlab2/policy/eval_result/images/comparison.png)

## Experimental Setup
As for DQfD, now a rule-based expert is used to generate demonstrations. The hyper-parameters mostly follows [Gordon-Hall et al.,
2020](https://arxiv.org/pdf/2004.08114.pdf) and shown as below:

|                    | Value         |
| -------------------|----------     |
| Steps              | 2,500,000     |
| Pre-training steps | 15,000        |
| Epsilon start      | 0.1           |
| Epsilon end        | 0.01          | 
| Epsilon decay rate | 500,000 steps |
| Discount factor    | 0.9           |
| Q network | 100d hidden layer and ReLU activation |
| Target network update period | 10,000 steps |
| Learning rate | 0.01 |
| L2 regularization weight  | 0.00001 |
| Max replay size | 100,000 |

Every 1,000 frames (steps), 3000 batches of size 32 would be sampled to train the model. 


## Experiment Result

|           | Task Success Rate |
| --------- | ----------------- |
| MLE       | 0.5157            |
| PPO       | 0.6136            |
| DQN      | 0.70         |
| DQfD_RE      | 0.77           |
| DQfD_NLE      | 0.74           |

The result is highly sensitive to hyper-parameter configurations.
