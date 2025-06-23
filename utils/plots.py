import matplotlib.pyplot as plt

def plot_results(records):
    iterations = [r['iteration'] for r in records]
    theta_norms = [np.linalg.norm(r['theta']) for r in records]
    w_norms = [np.linalg.norm(r['w']) for r in records]
    
    plt.plot(iterations, theta_norms, label='||theta||')
    plt.plot(iterations, w_norms, label='||w||')
    plt.xlabel('Iteration')
    plt.ylabel('Norm')
    plt.legend()
    plt.title('Two-Timescale RL Convergence')
    plt.show()
