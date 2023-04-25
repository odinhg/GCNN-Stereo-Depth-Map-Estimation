import numpy as np

class EarlyStopper():
    def __init__(self, limit = 10, delta= 0.01):
        self.limit = limit
        self.delta = delta 
        self.max_loss = np.inf 
        self.counter = 0
        print(f"Earlystopper active with limit: {self.limit} steps and delta: {self.delta}.")

    def __call__(self, loss):
        if loss < self.max_loss:
            self.max_loss = loss 
            self.counter = 0
        elif loss > self.max_loss + self.delta:
            self.counter += 1
            if self.counter >= self.limit:
                return True
        return False
