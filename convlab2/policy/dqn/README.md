# Deep Q-learning from Demonstrations
This algprithm takes a rule-based expert's actions as demonstrations. The state vector has 340 dimensions and action vector has 209 dimensions (each position in such a vector represent a single action). In order to introduce composite actions, 300 most common actions (including single and composite actions) in MultiWoz 2.1 dataset are mapped to the 209-dimension action space. Consequently, the total action number at each state is 300 and we can map every action to the 209-dimension act space by the action mapping file. 


