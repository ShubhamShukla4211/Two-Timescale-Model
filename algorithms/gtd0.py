import numpy as np
from .base_sa import TwoTimescaleSA
from .projection import sparse_project

class GTD0(TwoTimescaleSA):
    def __init__(self, feature_dim):
        super().__init__(feature_dim)
        self.A = np.eye(feature_dim)  # Placeholder
        self.b = np.zeros(feature_dim)  # Placeholder
    
    def update(self, phi, phi_next, reward, n):
        alpha = self.alpha_n(n)
        beta = self.beta_n(n)
        
        # Update w
        delta = reward + np.dot(self.theta, phi_next) - np.dot(self.theta, phi)
        self.w += beta * (delta * phi - self.w)
        
        # Update theta
        self.theta += alpha * (phi - phi_next) * np.dot(phi, self.w)
        
        # Sparse projection (if needed)
        self.theta = sparse_project(self.theta, n)
        self.w = sparse_project(self.w, n)
