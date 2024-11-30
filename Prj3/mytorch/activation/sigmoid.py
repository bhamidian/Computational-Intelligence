import numpy as np
from mytorch import Tensor, Dependency

def sigmoid(x: Tensor) -> Tensor:
    """
    TODO: implement sigmoid function
    hint: you can do it using function you've implemented (not directly define grad func)
    """
    
    return Tensor(np.ones(x.shape)) * (Tensor(np.ones(x.shape)) + (-x).exp()) ** -1