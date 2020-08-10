import numpy as np

class Loss:
    def __call__(self, predicted, actual):
        raise NotImplementedError
    def backward(self, predicted, actual):
        raise NotImplementedError

class MSE(Loss):
    def __call__(self, predicted, actual):
        return np.mean((predicted - actual)**2)
    
    def backward(self, predicted, actual):
        N = actual.shape[0]
        #print("predshape", predicted.shape, actual.shape)
        return (2./N)*(predicted - actual)

class BCE(Loss):
    def __call__(self, predicted, actual):
        acts = actual
        onemacts = 1 - actual
        preds = np.log(predicted)
        onempreds = np.log(1. - predicted)
        bcearray = - (acts*preds + onemacts*onempreds)
        return np.mean(bcearray)
    
    def backward(self, predicted, actual):
        N = actual.shape[0]
        #print("predshape", predicted.shape, actual.shape)
        return (1./N)*((predicted - actual)/(predicted*(1. - predicted)))
    
class CCE(Loss):    #Categorical Cross Entropy
    def __call__(self, predicted, actual):
        m = actual.shape[0]
        p = predicted
        log_likelihood = -np.log(p[range(m),y])
        loss = np.sum(log_likelihood) / m
        return loss
    
    def backward(self, predicted, actual):
        m = actual.shape[0]
        grad = predicted
        grad[range(m),y] -= 1
        grad = grad/m
        return grad