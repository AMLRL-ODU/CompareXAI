import matplotlib
matplotlib.use('Agg') # Force headless backend
import matplotlib.pyplot as plt
import numpy as np



def plot_noise_curves(results_dict, noise_levels, save_path, title_suffix=""):
    """
    Plots Accuracy and Probability curves vs Noise STD.
    Accepts an optional title_suffix to customize the chart title.
    """
    figure_names = ["Original", "Top", "Random", "Bottom"]
    colors = ['blue', 'red', 'green', 'orange']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Accuracy Curve
    for i, name in enumerate(figure_names):
        accuracies = []
        for std in noise_levels:
            data = np.array(results_dict[std])
            # Columns: [corr, prob, corr, prob...]
            col_idx = 2 * i 
            acc = np.mean(data[:, col_idx]) * 100
            accuracies.append(acc)
        axes[0].plot(noise_levels, accuracies, marker='o', label=name, color=colors[i])
        
    axes[0].set_title(f"Accuracy vs Noise Level {title_suffix}")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_xlabel("Noise Std Dev")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Probability Curve
    for i, name in enumerate(figure_names):
        probs = []
        for std in noise_levels:
            data = np.array(results_dict[std])
            # Columns: [corr, prob, corr, prob...]
            col_idx = 2 * i + 1
            avg_prob = np.mean(data[:, col_idx])
            probs.append(avg_prob)
        axes[1].plot(noise_levels, probs, marker='o', label=name, color=colors[i])

    axes[1].set_title(f"Mean True Class Probability vs Noise Level {title_suffix}")
    axes[1].set_ylabel("Probability")
    axes[1].set_xlabel("Noise Std Dev")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved Noise Plot: {save_path}")

def plot_occlusion_histograms(results, save_path, title_suffix=""):
    """
    Plots histograms of probabilities for occlusion analysis.
    Accepts an optional title_suffix to customize the chart title.
    """
    results = np.array(results)
    figure_names = ["Original", "Top Occlusion", "Random Occlusion", "Bottom Occlusion"]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i in range(4):
        ax = axes[i]
        # Probability columns: 1, 3, 5, 7
        prob_col = 2 * i + 1
        probs = results[:, prob_col]
        
        # Correctness columns: 0, 2, 4, 6
        corr_col = 2 * i
        acc = np.mean(results[:, corr_col]) * 100
        
        ax.hist(probs, bins=20, alpha=0.7)
        ax.axvline(np.mean(probs), color='red', linestyle='dashed')
        
        # Dynamic title with suffix
        title_text = f"{figure_names[i]}\nMean Prob: {np.mean(probs):.2f}, Acc: {acc:.1f}%"
        ax.set_title(title_text)
        ax.set_xlabel("Probability")
        ax.set_ylabel("Frequency")
        
    # Add a main title for the whole figure
    plt.suptitle(f"Occlusion Analysis {title_suffix}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved Occlusion Plot: {save_path}")