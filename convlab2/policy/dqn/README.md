# Deep Q-learning from Demonstrations
This algprithm takes a rule-based expert's actions as demonstrations. The state vector has 340 dimensions and action vector has 209 dimensions (each position in such a vector represent a single action). In order to introduce composite actions, Top 300 actions (including single and composite actions) in ConvLab-1 are mapped to the 209-dimension action space. Consequently, the total action number at each state is 319 (keep 209 single actions and add 110 composite actions from Top300) and we can map every action to the 209-dimension act space by the action mapping file. 

An siginificant issue is that the rule-based expert is designed upon ConvLab-2 and it can take many composite actions that are not in the 319 actions we get from ConvLab-1. This would lead to less available demonstrations for pre-train. For getting more demonstrations, now it is allowed to, for example, map an expert action \[Hotel-Inform-Addr, Hotel-Inform-Name, general-reqmore-none\] to \[Hotel-Inform-Addr, Hotel-Inform-Name\] in 319 actions.

Further issues: make multiprocessing applicable in training.
