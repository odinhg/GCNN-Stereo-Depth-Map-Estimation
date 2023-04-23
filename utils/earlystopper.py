class EarlyStopper():
    def __init__(self, limit = 10, delta= 0.01):
        self.limit = limit
        self.delta = delta 
        self.max_loss = 0 
        self.counter = 0
        print(f"Earlystopper active with limit: {self.limit} steps and delta: {self.delta}.")

    def __call__(self, loss):
        if loss < self.min_loss:
            self.min_loss = loss 
            self.counter = 0
        elif loss > self.min_loss + self.delta:
            self.counter += 1
            if self.counter >= self.limit:
                return True
        return False
