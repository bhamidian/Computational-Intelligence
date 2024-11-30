from mytorch import Tensor
import numpy as np

def MeanSquaredError(preds: Tensor, actual: Tensor):
    "TODO: implement Mean Squared Error loss"
    
    error = preds - actual
    error2 = error**2
    mse = error2
    size = Tensor(np.array([error2.data.size],dtype=np.float64))
    size = size**-1
    
    return mse * size
