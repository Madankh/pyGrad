from functools import partialmethod
import numpy as np

class Context:
    def __init__(self, arg, *tensors):
        self.arg = arg # output after forward prop
        self.parents = tensors # inputs that are used for computing arg or output
        self.saved_tensors = [] # empty list for saving a caches so we can used it when we deal with backprops


    # This method save_for_backward takes any number of arguments (*x) and extends (adds) 
    # them to the saved_tensors list. The extend method is used instead of append because
    # extend allows you to add all the elements of an iterable (like a list or tuple) to the 
    # existing list, whereas append would add the iterable itself as a single element.
        
    def save_for_backward(self, *x): # here *x mean we can pass any  number of arg using this 
        self.saved_tensors.extend(x)    # extend mean we append or insert all the data at ones rather then appending one by one using append method

class Tensor:
    def __init__(self, data):
        print(type(data), data)
        if type(data) != np.ndarray:
            print("error constructing tensor with %r" % data)
            assert(False)
        self.data = data
        self.grad = None

        # internal variables used for autograd graph constructtion
       self._ctx = None
    
    def __str__(self):
        return "Tensor %r with grad %r" % (self.data , self.grad)
    
    def backward(self, allow_fill = True):
        if self._ctx is None:
            return
        
        if self.grad is None and allow_fill:
            # fill in  the first grad with one
            
            assert self.data.size == 1
            self.grad = np.ones_like(self.data)
        assert(self.grad is not None)

        grads = self._ctx.arg.backward(self._ctx, self.grad)
        if len(self._ctx.parents) == 1:
            grads = [grads]
        for t,g in zip(self._ctx.parents, grads):
            if g.shape != t.data.shape:
                print("grad shape must match tensor shape in %r, %r != %r" % (self._ctx.arg, g.shape, t.data.shape))
                assert(False)
            t.grad = g
            t.backward(False)


class Function:
    def apply(self, arg, *x):
        ctx = Context() # This line creates a new instance of the Context class, which is used to store tensors required for the backward pass.
        x = [self]+list(x)
        ret = Tensor(arg.forward(ctx, *[t.data for t in x]))
        return ret
    
def register(name, fxn):
    setattr(Tensor, name, partialmethod(fxn.apply , fxn))

# **** Implement a few functions ****
class ReLU(Function):
    @staticmethod
    def forward(ctx , input):
        ctx.save_for_backward(input)
        return np.maximum(input , 0);

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.copy()
        grad_input[input < 0] = 0
        return grad_input
register('relu', ReLU)



class Dot(Function):
    @staticmethod
    def forward(ctx, input , weight):
        ctx.save_for_backward(input , weight)
        return input.dot(weight);
    
    @staticmethod
    def backward(ctx , grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_output.dot(weight.T)
        grad_weight = grad_output.dot(input)
        return grad_input, grad_weight
register('dot' , Dot)

class Add(Function):
    @staticmethod
    def forward(ctx, input , )

class Sum(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.sum()
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        return grad_output * np.ones_like(input)

register('sum' , Sum)

class LogSoftmax(Function):
    @staticmethod
    def forward(ctx, input):
        def log_sum_exp(x):
            c = x.max(axis=1)
            return c + np.log(np.exp(x - c.reshape((-1,1))).sum(axis=1))
        output = input - log_sum_exp(input)
        ctx.save_for_backward(output)
        return output

# class LogSoftmax(Function):
#     @staticmethod
#     def forward(ctx, input):
#         output = input - np.log(np.exp(input)).sum(axis=1))
#         ctx.save_for_backward(output)
#         return output
    
