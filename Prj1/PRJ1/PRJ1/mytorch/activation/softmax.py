import numpy as np
from mytorch import Tensor, Dependency


def softmax(x: Tensor) -> Tensor:
    """
    TODO: implement softmax function
    hint: you can do it using function you've implemented (not directly define grad func)
    hint: you can't use sum because it has not axis argument so there are 2 ways:
        1. implement sum by axis
        2. using matrix mul to do it :) (recommended)
    hint: a/b = a*(b^-1)
    """
    
    # soft max: exp(x) / sum(exp(x))
    
    exp = x.exp()
    sum_exp = exp @ np.ones((x.shape[1], 1))    # summing using _matmul
    
    return exp * (sum_exp ** -1) 
