import numpy as np

class TwoTimescaleSA:
    def __init__(self, feature_dim, alpha=0.6, beta=0.3):
        self.d = feature_dim
        self.theta = np.zeros(self.d)
        self.w = np.zeros(self.d)
        self.alpha = alpha
        self.beta = beta
    
    def alpha_n(self, n):
        return (n + 1) ** -self.alpha
    
    def beta_n(self, n):
        return (n + 1) ** -self.beta
    
    def get_theta(self):
        return self.theta
    
    def get_w(self):
        return self.w
