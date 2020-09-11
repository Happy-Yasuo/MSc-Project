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
Here is the evaluation result for the trained RoBERTa sequence classifier using Multiwoz 2.1 dataset:
|  |    |
| -------------------|----------     |
|    eval_loss       |   0.2517      |
| eval_accuracy             |  0.9056    |
| eval_f1 | 0.9093     |
| eval_precision | 0.8747     |
| eval_recall | 0.9469    |


For Taskmaster dataset:
|  |    |
| -------------------|----------     |
|    eval_loss       |  0.3088     |
| eval_accuracy             |  0.8613  |
| eval_f1 | 0.8661     |
| eval_precision | 0.8374      |
| eval_recall | 0.8969  |



For Personachat dataset:
|  |    |
| -------------------|----------     |
|    eval_loss       |   0.3618     |
| eval_accuracy             |  0.8389  |
| eval_f1 |      0.8339      |
| eval_precision |   0.8607    |
| eval_recall |  0.8087 |

For Scriptbase-j dataset:
|  |    |
| -------------------|----------     |
|    eval_loss       |   0.5008     |
| eval_accuracy             |   0.7421 |
| eval_f1 |        0.7286    |
| eval_precision |    0.7690   |
| eval_recall |  0.6923 |



Then we can train the DQfD_NLE by running `train.py` in the same directory. By setting '--data_set' with as 'taskmaster' or 'script', we can train DQfD_NLE using out-of-domain data. 
```bash
python train.py --data_set multiwoz
```

If we want to try RoFL, run
```bash
python train_RoFL.py --data_set taskmaster
python train_RoFL.py --data_set script
python train_RoFL.py --data_set personachat
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
| Learning rate decay rate| 0.99 |
| Minimum learning rate | 0.0000001|
| L2 regularization weight  | 0.00001 |
| Max replay size | 100,000 |

Every 1,000 frames (steps), 3000 batches of size 32 would be sampled to train the model. 

## Performance in terms of task domains
NLE Taskmaster

All 100 0.38                                                                      
restaurant 44 0.295                                              
taxi 17 0.294                                                        
hotel 39 0.205                                                     
train 32 0.312                                                                  
attraction 27 0.259                                                 
hospital 5 0.8                                                                    
police 2 0.5      

NLE Scriptbase

All 100 0.36                                                                      
restaurant 44 0.2727272727272727                                                  
taxi 17 0.4117647058823529                                                        
hotel 39 0.23076923076923078                                                      
train 32 0.3125                                                                   
attraction 27 0.4074074074074074                                                  
hospital 5 0.4                                                                    
police 2 0.5      

NLE Scriptbase fine tune

All 100 0.42                                                                      
restaurant 44 0.4318181818181818                                                  
taxi 17 0.29411764705882354                                                       
hotel 39 0.23076923076923078                                                      
train 32 0.40625                                                                  
attraction 27 0.4444444444444444                                                  
hospital 5 0.0                                                                    
police 2 0.5


NLE Personachat

All 100 0.42                 
restaurant 44 0.4090909090909091                                                  
taxi 17 0.23529411764705882                                                       
hotel 39 0.23076923076923078                                                      
train 32 0.3125                                                                    
attraction 27 0.3333333333333333                                                 
hospital 5 0.6                                                                    
police 2 0.5 

NLE Personachat fine tune

All 100 0.49                                                                      
restaurant 44 0.5                                                 
taxi 17 0.35294117647058826                                                      
hotel 39 0.3076923076923077                                                      
train 32 0.4375                                                                  
attraction 27 0.5185185185185185                                                 
hospital 5 0.4                                                                    
police 2 1.0
                                                     

## Experiment Result

|           | Task Success Rate | Average Reward |
| --------- | ----------------- | ----------------- |
| MLE       | 0.51            | |
| PPO       | 0.61            | |
| Rule Expert | 0.93            |13.70 |
| NLE Expert Multiwoz | 0.45            ||
| NLE Expert Taskmaster | 0.38            ||
| NLE Expert Personachat | 0.42            ||
| NLE Expert Script | 0.32            ||
| DQN      | 0.70         | |
| DQfD_RE      | 0.81           |
| DQfD_NLE      | 0.74           | 14.67|
| DQfD_NLE Taskmaster      | 0.56           | 12.66|
| DQfD_NLE Script     | 0.59           | 12.78|


