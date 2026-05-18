import torch
import numpy as np
import matplotlib
# Force headless mode for server execution
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import os
import gc
import sys
import copy


from xai_c.utils import get_model, get_weights_path, get_heatmap_fn
from xai_c.data import get_test_loader

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = '/home/ranap/decode_CNN/Dataset/Test/'
MODELS_TO_RUN = ['cnn', 'vgg']
METHODS_TO_RUN = ['LRP', 'PEEK', 'GradCAM', 'LIME'] 

# Analysis Parameters
TOP_K_PERCENT = 10      # Jaccard Threshold
WEIGHT_NOISE_STD = 0.05 # Standard deviation of noise added to weights (Small noise)

# --- HELPER FUNCTIONS ---

def add_noise_to_model(model, std=0.05):
    """
    Adds Gaussian noise to the model parameters in-place.
    """
    print(f"   -> Injecting weight noise (std={std})...")
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn_like(param) * std
            param.add_(noise)
    return model

def calculate_jaccard_top_k(map1, map2, top_k_percent=20):
    """
    Computes Jaccard Index (IoU) based on top k% salient pixels.
    """
    m1_flat = map1.flatten()
    m2_flat = map2.flatten()
    
    k = int(len(m1_flat) * (top_k_percent / 100.0))
    if k == 0: return 0.0

    # Determine thresholds
    thresh1 = np.partition(m1_flat, -k)[-k]
    thresh2 = np.partition(m2_flat, -k)[-k]
    
    # Binary Masks
    mask1 = m1_flat >= thresh1
    mask2 = m2_flat >= thresh2

    # Intersection over Union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0: return 0.0
    return intersection / union

def main():
    print(f"=== Starting Weight Robustness Analysis (2x2 Panel) ===")
    print(f"Device: {DEVICE}")
    print(f"Weight Noise Std: {WEIGHT_NOISE_STD}")
    print(f"Jaccard Threshold: Top {TOP_K_PERCENT}%")

    if not os.path.exists(DATA_PATH):
        print(f"Error: Data path {DATA_PATH} not found.")
        return
    
    # Batch size 1 required
    loader, _ = get_test_loader(DATA_PATH, batch_size=1)

    for model_name in MODELS_TO_RUN:
        print(f"\n" + "="*60)
        print(f"PROCESSING MODEL: {model_name.upper()}")
        print("="*60)

        torch.cuda.empty_cache()
        gc.collect()

        try:
            # 1. Load Original Model
            weights_path = get_weights_path(model_name)
            model = get_model(model_name, weights_path, DEVICE)
            
            # 2. Initialize Methods (Original)
            heatmap_fns = {
                m: get_heatmap_fn(m.lower(), model, DEVICE) for m in METHODS_TO_RUN
            }

            # --- PHASE 1: COMPUTE BASELINE MAPS ---
            print("\n[Phase 1] Computing Baseline Heatmaps (Original Weights)...")
            baseline_maps = [] # List of dicts: [{'LRP': map, 'PEEK': map...}, ...]
            
            for i, (img, _) in enumerate(tqdm(loader, desc="Baseline Pass")):
                img = img.to(DEVICE)
                img_maps = {}
                for method, fn in heatmap_fns.items():
                    try:
                        # Compute and store as numpy to save GPU memory
                        hm = fn(img).squeeze().detach().cpu().numpy()
                        img_maps[method] = hm
                    except Exception:
                        img_maps[method] = None # Mark failed
                baseline_maps.append(img_maps)

            # --- PHASE 2: PERTURB MODEL ---
            print("\n[Phase 2] Perturbing Model Weights...")
            # Note: We modify the model in-place. The 'heatmap_fns' wrappers 
            # hold a reference to this model, so they will automatically use the noisy weights now.
            add_noise_to_model(model, std=WEIGHT_NOISE_STD)

            # --- PHASE 3: COMPUTE NOISY MAPS & JACCARD ---
            print("\n[Phase 3] Computing Noisy Heatmaps & Comparing...")
            jaccard_results = [] # List of {'Method': ..., 'Jaccard': ...}

            for i, (img, _) in enumerate(tqdm(loader, desc="Robustness Pass")):
                img = img.to(DEVICE)
                base_maps = baseline_maps[i]
                
                for method, fn in heatmap_fns.items():
                    base_map = base_maps[method]
                    if base_map is None: continue # Skip if baseline failed

                    try:
                        # Compute Noisy Map
                        noisy_map = fn(img).squeeze().detach().cpu().numpy()
                        
                        # Calculate Jaccard
                        score = calculate_jaccard_top_k(base_map, noisy_map, TOP_K_PERCENT)
                        
                        jaccard_results.append({
                            'Method': method,
                            'Jaccard Index': score
                        })
                    except Exception:
                        pass # Skip if noisy gen failed

            # --- PHASE 4: PLOTTING (2x2 PANEL) ---
            if not jaccard_results:
                print("No results collected.")
                continue

            print(f"\n[Phase 4] Plotting Results for {model_name.upper()}...")
            df = pd.DataFrame(jaccard_results)
            
            # Calculate means for printing
            means = df.groupby('Method')['Jaccard Index'].mean()
            print("\n--- Mean Robustness Scores ---")
            print(means)

            # Create 2x2 Grid
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            
            methods_ordered = ['LRP', 'PEEK', 'GradCAM', 'LIME']
            # Define specific colors for consistency if desired
            palette = sns.color_palette("muted", n_colors=4)
            colors = dict(zip(methods_ordered, palette))

            for i, method in enumerate(methods_ordered):
                ax = axes[i]
                
                # Filter data for this method
                subset = df[df['Method'] == method]
                
                if subset.empty:
                    ax.text(0.5, 0.5, "No Data", ha='center')
                else:
                    # Plot Histogram
                    sns.histplot(
                        data=subset, 
                        x="Jaccard Index", 
                        bins=20, 
                        kde=True, # Optional: Add density curve
                        ax=ax, 
                        color=colors.get(method, 'blue'),
                        edgecolor="white"
                    )
                    
                    # Plot Mean Line (Red, Perpendicular)
                    mean_val = subset['Jaccard Index'].mean()
                    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2.5, label=f'Mean: {mean_val:.2f}')
                    ax.legend()

                # Formatting
                ax.set_title(method, fontsize=14, fontweight='bold')
                ax.set_xlim(0, 1.0)
                ax.set_xlabel("Jaccard Index")
                ax.set_ylabel("Frequency")
                ax.grid(True, alpha=0.3)

            plt.suptitle(f"Weight Robustness Distribution ({model_name.upper()})\n(Noise std={WEIGHT_NOISE_STD})", fontsize=16)
            
            if not os.path.exists("Final"): os.makedirs("Final")
            save_path = f"Final/weight_robustness_{model_name}_panel.png"
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"Plot saved to: {save_path}")

        except Exception as e:
            print(f"Critical Error for {model_name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()