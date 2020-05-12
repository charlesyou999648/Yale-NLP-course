=====================================
Chenyu You (cy346)
=====================================


------------------------------------------
PART A
------------------------------------------
Part A2:

1. What improvements did you try in the `myRegression` class?  Describe the structure. And do you think it improved?
>>>	I add one linear layer to improve the network robustness. The preditions match the true data very well.

2. Do you think only using `nn.Linear` layers can learn this function: $y=sin(x)$? Why or why not?
>>>	No. The universal approximation theorem states that a feed-forward network with a single hidden layer containing a finite number of neurons can approximate continuous functions on compact subsets of Rn, under mild assumptions on the activation function. Clearly, just the linear part of NNs won't be able to perform this task since it is linear. If we add activation function after `nn.Linear` layers, it can learn the function $y=sin(x)$.

