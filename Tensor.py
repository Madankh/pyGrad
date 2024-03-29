from functools import partialmethod
import numpy as np

class Context:
    def __init__(self):
        self.saved_tensors = []
    def save_for_backward(self, *x):
        self.saved_tensors.extend(x)    

class Tensor:
    def __init__(self, data, _children=()):
        self.data = data
        self.grad = np.zeros_like(data)

        # internal variables used for autograd graph constructtion
        self._prev = set(_children)