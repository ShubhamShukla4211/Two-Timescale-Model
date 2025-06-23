import numpy as np

class SimpleMDP:
    def __init__(self):
        self.feature_dim = 5
        self.gamma = 0.9
    
    def sample_transition(self):
        phi = np.random.uniform(-1, 1, self.feature_dim)
        phi_next = np.random.uniform(-1, 1, self.feature_dim)
        reward = np.random.uniform(-1, 1)
        return phi, phi_next, reward
