import numpy as np
import scipy.stats
'''
gda = gaussian discriminant analysis

input:
X: list of n-features
y: class

should be generalized to n-dimensions??? not sure though???
'''

class GDA:
    def __init__(self):
        '''
        example mean:
        [[mean of x of class 1, mean of y of class 1]
         [mean of x of class 2, mean of y of class 2]]
        '''
        self.means = []
        self.stds = []

    def fit(self, X, y):
        idx = []

        # finds index of each class
        for i in np.unique(y):
            idx.append([j for j,k in enumerate(y) if k == i])
        
        for i in idx:
            m = []
            s = []
            for j in range(len(X[0])):
                m.append(X[i,j].mean())
                s.append(X[i,j].std())
            self.means.append(m)
            self.stds.append(s)

    def predict(self, X):
        # example input: [x,y]

        L = []
        for m, s in zip(self.means, self.stds):
            l = 0
            for idx,i in enumerate(X):
                l += np.log(scipy.stats.norm(m[idx], s[idx]).pdf(i))
            L.append(l)
        
        # since we are not taking the negative of the log the maximum value has the least loss
        return np.argmax(L)