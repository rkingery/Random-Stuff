"""
Class for implementing arbitrary an feedforward neural network
Notation mostly from Andrew Ng's deeplearning.ai courses

TODO: Implement gradient checking and early stopping, 
      expand and organize optimization algorithms (batch,SGD,adam,etc),
      fix dW_eff,db_eff in adam (numerical division issues...early stop?)

Author: Ryan S. Kingery
Last Updated: Aug 27, 2017
"""

import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork(object):
    """
    Creates a feedforward neural network instance
    Parameters:
        layers: list of layer sizes [input layer size, ..., output layer size]
        activation: type of activation function for all but last layer,
                    choices are 'relu','sigmoid', 'tanh', 'linear'
                    default = 'relu'
        last_activation: type of activation function for last layer,
                         choices the same as for activation
                         default = 'sigmoid'
        initialization: factor used to initialize weight and bias parameters,
                        choices are 'He', 'Xavier', 'Bengio', or any input number
                        default = 'He'
        dropout_prob: if drop_out prob is False this does nothing, otherwise it
                      implements dropout with keep probability = dropout_prob,
                      value must be False or number between 0 and 1
                      default = False
        alpha: initial learning rate for gradient descent
               default = 0.1
        learning_rate: rate decay parameter for gradient descent
                       default = 0.0 (no decay)
        batch_size: batch size for SGD
                    default = 64
        iters: number of iterations/epochs to run SGD
               default = 1000
        reg_param: regularization parameter (lambda)
                   default = 0.001
        momentum: momentum parameter (beta)
                  default = 1.0 (no momentum correction)
        rmsprop: RMSprop parameter
                 default = 1.0 (no RMSprop correction)
        decay_rate: decay rate for learning rate alpha
                    default = 0.0 (no rate decay)
        gradient_checking: implements gradient checking if True, not if False
                           default = False
        print_loss: if True, prints loss for each iteration, not if False
                    default = False
        loss_plot: if True, plots loss over iterations, not if False
                   default = False                   
    """
    def __init__(self,layers,activation='relu',last_activation='sigmoid',
                 initialization='He',dropout_prob=False,alpha=0.1,learning_rate=0.0, \
                 batch_size=64,epochs=1000,reg_param=0.001,momentum=0.9,rmsprop=0.999, \
                 decay_rate=0.0,gradient_checking=False,print_loss=False,loss_plot=False):
        """
        Initializer for the neural network
        """
        self.layers = layers
        self.parameters = {}
        self.activation = activation
        self.last_activation = last_activation
        self.initialization = initialization
        self.dropout_prob = dropout_prob
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.reg_param = reg_param
        self.momentum = momentum
        self.rmsprop = rmsprop
        self.decay_rate = decay_rate
        self.gradient_checking = gradient_checking
        self.print_loss = print_loss
        self.loss_plot = loss_plot
        self.L = len(self.layers)-1
        for l in range(1, self.L+1):
            self.parameters['W'+str(l)] = np.random.randn(self.layers[l],self.layers[l-1]) * \
                                          self._initialize(l)
            self.parameters['b'+str(l)] = np.zeros((self.layers[l],1))
        
    def fit(self,X,y):
        """
        Train inputs X with labels y using gradient descent on given neural network.
        Parameters:
            X: design array with shape (features, examples)
            y: target array with shape (1, examples)
        Returns: List of loss values over each epoch if print_loss is True, else None
        """
        loss_grid = []
        for epoch in range(self.epochs):
            if self.gradient_checking == True:
                pass
                #grads_approx = self._gradient_checking()
                # print something
            yhat = self._adam(X,y,epoch)
            loss = self._loss(X.shape[1],y,yhat)
            loss_grid += [loss]
            if self.print_loss is True:
                #print "Loss at iteration "+str(epoch)+": "+str(loss)
                return loss_grid
        if self.loss_plot is True:
            plt.plot(loss_grid)#,linestyle='none',marker='.')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.title('Loss During Gradient Descent')
            plt.show()
               
    def predict(self,X):
        """Returns: predicted labels y given input matrix X"""
        inputs,activations = self._feedforward(X)
        return(activations["A"+str(self.L)])
    
    def score(self,X,y):
        """Returns: accuracy score of predicted values wrt actual labels y"""
        y = np.round(y)
        yhat = np.round(self.predict(X))
        return float(sum(np.equal(y,yhat)[0,:])) / X.shape[1]
    
    def _initialize(self,l):
        """Private: returns initializing factor for layer l"""
        if self.initialization == 'He':
            return np.sqrt(2./self.layers[l-1])
        if self.initialization == 'Xavier':
            return np.sqrt(1./self.layers[l-1])
        if self.initialization == 'Bengio':
            return np.sqrt(2./(self.layers[l-1]+self.layers[l]))
        if type(self.initialization) is not str:
            return self.initialization
    
    def _activation(self,Z,string):
        """Private: returns activation function of type string with input array Z"""
        if string == 'relu':
            return np.maximum(0,Z)
        if string == 'sigmoid':
            return 1./(1+np.exp(-Z))
        if string == 'tanh':
            return np.tanh(Z)
        if string == 'linear':
            return Z
        
    def _activation_derivative(self,Z,string):
        """Private: returns activation function derivative of type string with input array Z"""
        if string == 'relu':
            return np.maximum(np.zeros(Z.shape),np.ones(Z.shape))
        if string == 'sigmoid':
            sigmoid = lambda x : 1./(1+np.exp(-x))
            return sigmoid(Z)*(1-sigmoid(Z))
        if string == 'tanh':
            return 1-np.tanh(Z)**2
        if string == 'linear':
            return 1+np.zeros(Z.shape)
    
    def _loss(self,m,y,yhat):
        """Private: returns log loss / cross entropy funtion"""
        reg_list = [np.linalg.norm(self.parameters['W'+str(l)],'fro')**2 \
                    for l in range(1,self.L+1)]
        return -1./m*(y*np.log(yhat)+(1-y)*np.log(1-yhat)).sum(axis=1)[0] \
               + 1./(2*m)*self.reg_param*sum(reg_list)
    
    def _feedforward(self,X):
        """Private: feedforward process that returns all network input values as 2 dictionaries"""
        inputs = {}
        activations = {'A0':X}
        for l in range(1,self.L+1):
            inputs['Z'+str(l)] = np.dot(self.parameters['W'+str(l)],activations['A'+str(l-1)]) \
                                 + self.parameters['b'+str(l)]
            activations['A'+str(l)] = self._activation(inputs['Z'+str(l)],self.activation)
        inputs['Z'+str(self.L)] = \
            np.dot(self.parameters['W'+str(self.L)],activations['A'+str(self.L-1)]) \
            + self.parameters['b'+str(self.L)]
        activations['A'+str(self.L)] = \
            self._activation(inputs['Z'+str(self.L)],self.last_activation)
        return inputs,activations
                
    def _backprop(self,m,y,inputs,activations):
        """Private: backpropagation algorithm returning all weight/bias gradients of the loss"""
        AL = activations["A"+str(self.L)]
        dAL = -(np.divide(y,AL) - np.divide(1-y,1-AL))
        dZL = dAL*self._activation_derivative(inputs['Z'+str(self.L)],self.last_activation)
        grads = {'dA'+str(self.L):dAL,'dZ'+str(self.L):dZL}
        for l in range(self.L,0,-1):
            A_prev = activations['A'+str(l-1)]
            dZ = grads['dZ'+str(l)]
            W = self.parameters['W'+str(l)]
            grads['dA'+str(l-1)] = np.dot(W.T,dZ)
            if l >= 1:
                grads['dW'+str(l)] = 1./m*np.dot(dZ,A_prev.T)
                grads['db'+str(l)] = 1./m*np.sum(dZ,axis=1,keepdims=True)
            if l > 1:
                dA_prev = grads['dA'+str(l-1)]
                Z_prev = inputs['Z'+str(l-1)]
                grads['dZ'+str(l-1)] = dA_prev*self._activation_derivative(Z_prev,self.activation)
        return grads
    
    def _adam(self,X,y,epoch):
        """
        Private: Performs adam optimization for 1 iteration / epoch
        Returns: Output activation yhat after num_batches stochastic updates
        """
        assert self.batch_size <= X.shape[1], "Batch size must be no larger than number of inputs"
        num_batches = X.shape[1] // self.batch_size
        m = self.batch_size
        X_list,y_list = self._partition(X,y,m,num_batches)
        for t in range(1,num_batches+1):
            # update X_t, y_t situation first
            inputs,activations = self._feedforward(X_list[t-1])
            grads = self._backprop(m,y_list[t-1],inputs,activations)
            for l in range(1,self.L+1):
                W, b = (self.parameters["W"+str(l)],self.parameters["b"+str(l)])
                dW,db = (grads["dW"+str(l)],grads["db"+str(l)])
                dW_mom,db_mom = (np.zeros(dW.shape),np.zeros(db.shape))
                dW_mom = self.momentum*dW_mom + (1-self.momentum)*dW
                dW_mom /= (1-self.momentum**t)
                db_mom = self.momentum*db_mom + (1-self.momentum)*db
                db_mom /= (1-self.momentum**t)
                dW_rms,db_rms = (np.zeros(dW.shape),np.zeros(db.shape))
                dW_rms = self.rmsprop*dW_rms + (1-self.rmsprop)*np.power(dW,2)
                dW_rms /= (1-self.rmsprop**t)
                db_rms = self.rmsprop*db_rms + (1-self.rmsprop)*np.power(db,2)
                db_rms /= (1-self.rmsprop**t)
                dW_eff = dW_mom#np.divide(dW_mom,np.sqrt(dW_rms)+10e-8)
                db_eff = db_mom#np.divide(db_mom,np.sqrt(db_rms)+10e-8)
                alpha_eff = self.alpha/(1+self.decay_rate*epoch)
                W -= alpha_eff*(dW_eff + self.reg_param/m*W)
                b -= alpha_eff*(db_eff + self.reg_param/m*b)
                self.parameters["W"+str(l)] = W
                self.parameters["b"+str(l)] = b
        inputs,activations = self._feedforward(X)
        yhat = activations["A"+str(self.L)]
        return yhat
    
    def _partition(self,X,y,m,num_batches):
        """
        Private: Partitions X,y randomly into num_batches examples
        Returns: X_list of partioned X examples, y_list of partitioned y examples
        """
        X_list = []
        y_list = []
        M = X.shape[1]
        perm = list(np.random.permutation(M))
        X = X[:,perm]
        y = y[:,perm].reshape((1,M))
        for k in range(0,num_batches):
            mini_batch_X = X[:,k*m:(k+1)*m]
            mini_batch_y = y[:,k*m:(k+1)*m]
            X_list.append(mini_batch_X)
            y_list.append(mini_batch_y)
        if M % m != 0:
            mini_batch_X = X[:,num_batches*m:]
            mini_batch_y = y[:,num_batches*m:]
            X_list.append(mini_batch_X)
            y_list.append(mini_batch_y)
        return X_list,y_list
        
#    def _gradient_checking(self,m,y,inputs,activations):
#        """
#        Private: Calculates numerical gradients of loss to check accuracy of _backprop()
#        NOT FINISHED
#        """
#        eps = 10e-6
#        grads_approx = {}
#        yhat = activations['A'+str(self.L)]
#        loss = self._loss(m,y,yhat)
#        for l in range(1,self.L+1):
#            grads_approx['dW'+str(l)] = 0
#            grads_approx['db'+str(l)] = 0
#        return grads_approx
        
    
if __name__ == '__main__':
    X = np.random.rand(10,100)
    y = np.random.rand(1,100)
    net = NeuralNetwork([10,10,5,5,1],epochs=2000,batch_size=20,loss_plot=True)
    net.fit(X,y)
    print net.score(X,y)