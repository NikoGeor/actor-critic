import argparse
import numpy as np
from dataclasses import dataclass
from mdptoolbox.example import forest
from mdptoolbox.mdp import ValueIteration

from src import MDP, ActorCriticAlgorithm

# 1. Define your Configuration Class
@dataclass
class Config:
    neurons: int = 1000
    T: int = 100
    alpha: float = 25.0
    zeta: float = 25.0
    beta: float = 1.0
    num_experiments: int = 30
    discount_factor: float = 0.7

def parse_args() -> Config:
    """Parses command line arguments and returns a Config object."""
    parser = argparse.ArgumentParser(description="Run JAX Actor-Critic Experiments")
    
    # We set the default values to match the Config class
    parser.add_argument("--neurons", type=int, default=1000, help="Number of neurons (N)")
    parser.add_argument("--t", type=int, default=100, help="Time horizon (T)")
    parser.add_argument("--alpha", type=float, default=25.0, help="Critic learning rate scaling")
    parser.add_argument("--zeta", type=float, default=25.0, help="Actor learning rate scaling")
    parser.add_argument("--beta", type=float, default=1.0, help="Scaling exponent")
    parser.add_argument("--runs", type=int, default=30, help="Number of experiments to run")
    
    args = parser.parse_args()
    
    # Map the parsed arguments into our Config class
    return Config(
        neurons=args.neurons,
        T=args.t,
        alpha=args.alpha,
        zeta=args.zeta,
        beta=args.beta,
        num_experiments=args.runs
    )

def main(cfg: Config):
    print(f"Running with configuration: {cfg}")
    
    P, R = forest(S=3, r1=4, r2=2, p=0.1)
    states = np.asarray([[0], [1], [2]])
    actions = np.asarray([[0], [1]])
    init_dist = np.asarray([1, 0, 0])
    
    mdp = MDP(states, actions, init_dist, R, P.transpose([1, 0, 2]), discount=cfg.discount_factor)

    print("Computing optimal policy baseline...")
    vi = ValueIteration(
        transitions=P, reward=R, discount=cfg.discount_factor, 
        epsilon=1e-10, max_iter=100000, initial_value=0
    )
    vi.run()

    optimal_policy = np.zeros((states.shape[0], actions.shape[0]))
    for state_id in range(states.shape[0]):
        optimal_policy[state_id, vi.policy[state_id]] += 1

    critic_losses, actor_losses = [], []
    bellman_losses, critic_dists, rewards_hist = [], [], []
    Qs, pis = [], []

    # 2. Use the cfg object for your hyperparameters
    print(f"Starting {cfg.num_experiments} experiments...")
    for i in range(cfg.num_experiments):
        print(f"\n--- Experiment {i+1}/{cfg.num_experiments} ---")
        
        AC = ActorCriticAlgorithm(
            N=cfg.neurons, 
            T=cfg.T, 
            beta=cfg.beta, 
            alpha=cfg.alpha, 
            zeta=cfg.zeta, 
            mdp=mdp,
            seed=i  
        )
        
        AC.fit(optimal_policy, mdp)
        
        critic_losses.append(AC.critic_losses)
        actor_losses.append(AC.actor_losses)
        Qs.append(AC.Q)
        pis.append(AC.pi)
        bellman_losses.append(AC.bellman_losses)
        critic_dists.append(AC.critic_dists)
        rewards_hist.append(AC.rewards)

    # 3. Dynamically name the output files so they don't overwrite each other
    suffix = f"N{cfg.neurons}_T{cfg.T}_A{int(cfg.alpha)}"
    print(f"\nSaving results to .npy files with suffix {suffix}...")
    
    np.save(f'critic_losses_{suffix}.npy', np.asarray(critic_losses))
    np.save(f'actor_losses_{suffix}.npy', np.asarray(actor_losses))
    np.save(f'Qs_{suffix}.npy', np.asarray(Qs))
    np.save(f'pis_{suffix}.npy', np.asarray(pis))
    np.save(f'bellman_losses_{suffix}.npy', np.asarray(bellman_losses))
    np.save(f'critic_dists_{suffix}.npy', np.asarray(critic_dists))
    np.save(f'rewards_{suffix}.npy', np.asarray(rewards_hist))

if __name__ == "__main__":
    # Parse terminal arguments and pass them to main()
    config = parse_args()
    main(config)