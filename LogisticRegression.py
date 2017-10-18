import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression(object):
    """
    Implement logistic regression model with input matrix X, target y
    Parameters:
        X: design array with shape (features, examples)
        y: target array with shape (1, examples)
    """
    def __init__(self,X,y):
        self.X = X
        self.y = y
        self.n = np.shape(X)[0]     # number of features
        self.m = np.shape(X)[1]     # number of data points
        self.w = np.zeros((1,self.n))
        self.b = 0
        
    def train(self,alpha=0.1,iters=1000,loss_plot=False):
        """
        Train model using gradient descent with learning rate alpha over iters iterations
        If loss_plot = True, also prints plot of loss change over training iterations
        """
        Jgrid = []
        for i in range(iters):
            Z = np.dot(self.w,self.X) + self.b
            A = self._sigmoid(Z)
            dZ = A - self.y
            dw = 1./self.m * np.dot(dZ,self.X.T)
            db = 1./self.m * np.sum(dZ)
            self.w -= alpha*dw
            self.b -= alpha*db
            Jgrid += [self._log_loss(A)]
        if loss_plot is True:
            plt.plot(Jgrid,linestyle='none',marker='.')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.title('Loss During Gradient Descent')
            plt.show()
        #return self.w,self.b
    
    def predict(self,X):
        """Returns predicted target values of inputed design matrix X"""
        Z = np.dot(self.w,self.X) + self.b
        return self._sigmoid(Z)
    
    def score(self,X):
        """Returns accuracy score of model on inputed design matrix X"""
        self.y = np.round(self.y)
        yhat = np.round(self.predict(X))
        return float(sum(np.equal(self.y,yhat)[0,:])) / self.m
            
    def _sigmoid(self,z):
        """Private: Returns sigmoid function in vectorized format"""
        return 1./(1+np.exp(-z))
    
    def _log_loss(self,yhat):
        """Private: Returns log loss L(y,yhat) given predicted targets yhat"""
        return 1./self.m*(-self.y*np.log(yhat)-(1-self.y)*np.log(1-yhat)).sum(axis=1)[0]