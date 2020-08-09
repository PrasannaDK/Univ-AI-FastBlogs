import numpy as np

class Learner:
    
    def __init__(self, loss, model, opt, epochs, X_train, y_train, X_test, y_test):
        self.loss = loss
        self.model = model
        self.opt = opt
        self.epochs = epochs
        self.cbs = []
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
    def set_callbacks(self, cblist):
        for cb in cblist:
            self.cbs.append(cb)
            
    def __call__(self, cbname, *args):
        status = True
        for cb in self.cbs:
            cbwanted = getattr(cb, cbname, None)
            status = status and cbwanted and cbwanted(*args)
        return status
    
    def train_loop(self, dl):
        self.dl = dl # dl added in here
        bs = self.dl.bs
        datalen = len(self.dl.data)
        self.bpe = datalen//bs
        self.afrac = 0.
        if datalen % bs > 0:
            self.bpe  += 1
            self.afrac = (datalen % bs)/bs
        self('fit_start')
        for epoch in range(self.epochs):
            self('epoch_start', epoch)
            for inputs, targets in dl:
                self("batch_start", dl.current_batch)
                
                # make predictions
                predicted = self.model(inputs)
                #print('predicted- ', predicted.shape)

                # actual loss value
                epochloss = self.loss(predicted, targets)
                self('after_loss', epochloss)

                # calculate gradient
                intermed = self.loss.backward(predicted, targets)
                self.model.backward(intermed)

                # make step
                self.opt.step(self.model)              
                
                self('batch_end')
            self('epoch_end')
            #validate the model for each epoch
            validateTrain = np.mean((self.model(self.X_train) >= 0.5)*1 == self.y_train)
            validateTest = np.mean((self.model(self.X_test) >= 0.5)*1 == self.y_test)
            self('epoch_accuracies', validateTrain, validateTest)
            
        self('fit_end')
        return epochloss