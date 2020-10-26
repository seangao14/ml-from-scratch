import numpy as np
from tqdm import tqdm

class LogisticRegression:
    def __init__(self, lr=0.01, epochs=10000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def fit(self, X, y):
        '''
        math in this code is from here:
        http://cs230.stanford.edu/fall2018/section_files/section3_soln.pdf
        '''
        # initialize weights and bias to 0
        self.weights = np.zeros((X.shape[1]))
        self.bias = 0

        for i in tqdm(range(self.epochs)):
            pred = self.sigmoid(np.dot(X,self.weights) + self.bias)

            # no need to compute cost
            # backprop:
            dw = np.dot(X.T, (pred-y))
            db = np.sum(pred-y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db