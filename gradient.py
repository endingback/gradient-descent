import numpy as np
import pandas as pd

class Regression:
    def __init__(self, data):
        self.x = np.mat(data.iloc[:,:-1].values)
        self.y = np.mat(data.iloc[:, -1].values)
        self.data = data

    def BGD(self, lr, iternum):
        theat = np.mat(np.zeros(self.x.shape[1], 1))
        for _ in range(iternum):
            grad = 2 / self.x.shape[0] * (self.x.T * (self.x * theat - self.y))
            theat = theat - grad * lr
        return np.ravel(theat)

    def SGD(self, lr, iternum):
        theat = np.mat(np.zeros(self.x.shape[1], 1)
        for _ in range(iternum):
            rnd = np.random.randint(self.x.shape[0])
            grad = 2 / self.x.shape[0] * (self.x[rnd] * (self.x[rnd] * theat[rnd] - self.y[rnd]))
            theat = theat - grad * lr
        return np.ravel(theat)

    def mini_batch(self, lr, iternum):
        theat = np.mat(np.zeros(self.x.shape[1], 1))
        all_data = pd.DataFrame(self.x)
        for _ in range(iternum):
            batch_index = all_data.sample(frac = 0.1, replace = False).index #未放回抽样
            x_mini = self.x[batch_index, :]
            y_mini = self.y[batch_index, :]
            grad = 2 / self.x.shape[0] * (x_mini * (x_mini * theat[batch_index] - y_mini))
            theat = theat - grad * lr
        return np.ravel(theat)    
