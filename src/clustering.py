import numpy as np


'''
X: list of 2d arrays
fit() will:
    define centroids, assign y value (0 or 1) to each X
'''
class K_means_cluster:
    def __init__(self):
        self.data = None
        self.clss = None
        self.c1 = None
        self.c2 = None
        self.idx0 = None
        self.idx1 = None
    
    def fit(self, X):
        self.data = X

        xs, ys = X[:,0], X[:,1]
        # init centroids:
        x_range = xs.max() - xs.min()
        y_range = ys.max() - ys.min()

        self.c1 = [np.random.random()*x_range + xs.min(), np.random.random()*y_range + ys.min()]
        self.c2 = [np.random.random()*x_range + xs.min(), np.random.random()*y_range + ys.min()]

        while True:
            self.get_clss()
            new_c1, new_c2 = self.new_c()
            if (new_c1, new_c2) == (self.c1, self.c2) or (new_c1, new_c2) == (self.c2, self.c1):
                break
            self.c1, self.c2 = new_c1, new_c2

    # classify each sample based on current centroids
    def get_clss(self):
        # list of distances from c1 and c2
        d_c1 = np.linalg.norm(self.data-self.c1, axis=1)
        d_c2 = np.linalg.norm(self.data-self.c2, axis=1)
        
        self.clss = d_c1<d_c2

    # calculate new centroids
    def new_c(self):
        self.idx0 = [i for i,j in enumerate(self.clss) if j == 0]
        self.idx1 = [i for i,j in enumerate(self.clss) if j == 1]

        c1 = (self.data[self.idx0,0].mean(), self.data[self.idx0,1].mean())
        c2 = (self.data[self.idx1,0].mean(), self.data[self.idx1,1].mean())

        return c1, c2