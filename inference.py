import numpy as np
import matplotlib.pyplot as plt
import os

def plot_metric(metric_name, title, ylabel, betas, colors):
    plt.figure(figsize=(8, 6))
    x = np.arange(0, 1, 0.01)  # 100 evaluation steps representing training progress
    
    for beta, color in zip(betas, colors):
        # Adjust this filename format to match exactly how you saved them in run_experiments.py
        # For example, if you saved them as 'rewards_N1000_B0.500.npy'
        filename = f"{metric_name}_N1000_T100_B{beta:.3f}.npy"
        
        if os.path.exists(filename):
            data = np.load(filename)
            
            # Since data is shaped (30_runs, 100_evals), we average across the 30 runs (axis=0)
            if data.ndim == 2:
                mean_data = np.mean(data, axis=0)
            else:
                mean_data = data
            
            plt.plot(x, mean_data, color=color, label=f'β = {beta:.3f}')
        else:
            print(f"Warning: Could not find {filename}")

    plt.xlabel('Training Progress')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure as a PNG and close the plot to free memory
    output_filename = f"{metric_name}_comparison.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully generated and saved {output_filename}")

def main():
    # Define the configurations
    betas = [0.500, 0.625, 0.750, 0.875, 1.000]
    colors = ['red', 'orange', 'yellow', 'green', 'blue']
    
    # Define the specific metrics to plot
    metrics = [
        {'name': 'rewards', 'title': 'Expected Discounted Future Rewards', 'ylabel': 'Rewards'},
        {'name': 'critic_losses', 'title': 'Critic Training Loss', 'ylabel': 'Loss'},
        {'name': 'actor_losses', 'title': 'Actor Training Loss', 'ylabel': 'Loss'},
        {'name': 'bellman_losses', 'title': 'Bellman Error', 'ylabel': 'Loss'},
        {'name': 'critic_dists', 'title': 'Distance to Optimal Q-Function', 'ylabel': 'Distance'}
    ]
    
    print("Generating plots...")
    for m in metrics:
        plot_metric(
            metric_name=m['name'],
            title=m['title'],
            ylabel=m['ylabel'],
            betas=betas,
            colors=colors
        )

if __name__ == "__main__":
    main()