import numpy as np

class Environment():

    def __init__(self, size, num_points, T_0):
        ''' the environment is simple the interval: [0, self.size] '''
        self.size = size
        self.T_0 = T_0 # temp at origin
        self.env = np.linspace(start = 0, stop = size, num = num_points)

    def temperature(self, x):
        return self.T_0 / (x ** 2 + 1)
