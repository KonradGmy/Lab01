import random as rand

MAX_ERAS = 100


class Neuron:
    def __init__(self, eta, W_range, theta_range):
        self.theta = rand.random() * (theta_range[0] - theta_range[1]) + theta_range[1]
        self.eta = eta
        self.W = None
        self.z = 0
        self.eras = 0
        self.W_range = W_range
        self.theta_range = theta_range

    def fit(self, X, y):
        pass

    def actual_y(self, X):
        pass

    def actualize_W(self, error, X):
        pass

    def show(self, X):
        pass


class Adaline(Neuron):
    def __init__(self, eta, W_range, theta_range, threshold):
        super().__init__(eta, W_range, theta_range)
        self.threshold = threshold

    def fit(self, X, y):
        self.W = [rand.random() * (self.W_range[0] - self.W_range[1]) + self.W_range[1] for x in X[0]]
        mean_squared_error = self.threshold
        self.eras = 0
        while mean_squared_error >= self.threshold:
            mean_squared_error = 0
            self.eras += 1
            if self.eras >= MAX_ERAS:
                return
            for x, y_ in zip(X, y):
                act_y = self.actual_y(x)
                error = y_ - act_y
                mean_squared_error += error ** 2
                self.actualize_W(error, x)
            mean_squared_error /= len(X)

    def actual_y(self, X):
        self.z = 0
        for x, w in zip(X, self.W):
            self.z += x * w
        return self.z + self.theta

    def actualize_W(self, error, X):
        for i in range(len(self.W)):
            self.W[i] += self.eta * error * X[i]
        self.theta += self.eta * error

    def show(self, X):
        for x in X:
            print(x, 1 if self.actual_y(x) > 0 else -1)


class Perceptron(Neuron):
    def __init__(self, eta, W_range, theta_range, unipolar=True):
        super().__init__(eta, W_range, theta_range)
        self.unipolar = unipolar

    def fit(self, X, y):
        self.W = [rand.random() * (self.W_range[0] - self.W_range[1]) + self.W_range[1] for x in X[0]]
        error_flag = True
        self.eras = 0
        while error_flag:
            error_flag = False
            self.eras += 1
            if self.eras >= MAX_ERAS:
                return
            for x, y_ in zip(X, y):
                act_y = self.actual_y(x)
                error = y_ - act_y
                if error != 0:
                    error_flag = True
                self.actualize_W(error, x)

    def actual_y(self, X):
        self.z = 0
        for x, w in zip(X, self.W):
            self.z += x * w
        self.z += self.theta
        if self.unipolar:
            return 1 if self.z > 0 else 0
        return 1 if self.z > 0 else -1

    def actualize_W(self, error, X):
        for i in range(len(self.W)):
            self.W[i] += self.eta * error * X[i]
        self.theta += self.eta * error

    def show(self, X):
        for x in X:
            print(x, self.actual_y(x))