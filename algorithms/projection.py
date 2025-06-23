import numpy as np

def sparse_project(x, n, R=10):
    """Sparse projection when n is of form k^k - 1"""
    import math
    if n > 1:
        k = math.log(n + 1) / math.log(math.log(n + 1))
        if abs(n - (k ** k - 1)) < 1e-3:
            norm = np.linalg.norm(x)
            if norm > R:
                x = x * (R / norm)
    return x
