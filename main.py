from algorithms.gtd0 import GTD0
from envs.simple_mdp import SimpleMDP
from utils.logger import Logger
from utils.plots import plot_results

def main():
    # Initialize environment and data sampler
    mdp = SimpleMDP()
    
    # Initialize GTD(0)
    gtd0 = GTD0(mdp.feature_dim)
    
    # Initialize logger
    logger = Logger()
    
    # Run the algorithm
    num_iters = 10000
    for n in range(1, num_iters + 1):
        phi, phi_next, reward = mdp.sample_transition()
        gtd0.update(phi, phi_next, reward, n)
        
        if n % 100 == 0:
            logger.log(n, gtd0.get_theta(), gtd0.get_w())
    
    # Plot the results
    plot_results(logger.records)

if __name__ == "__main__":
    main()
