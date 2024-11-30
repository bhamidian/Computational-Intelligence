from mytorch import Tensor
import numpy as np
from mytorch.activation import softmax

def CategoricalCrossEntropy(preds: Tensor, label: Tensor):
    "TODO: implement Categorical Cross Entropy loss"
    
    size = Tensor(np.ones(preds.shape) / label.shape[0])
    
    return (label * softmax(preds)).sum() * size