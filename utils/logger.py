class Logger:
    def __init__(self):
        self.records = []
    
    def log(self, n, theta, w):
        self.records.append({
            'iteration': n,
            'theta': theta.copy(),
            'w': w.copy()
        })
