import matplotlib
matplotlib.use('Agg')
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import os
import gc
import sys

# --- SIMPLIFIED IMPORTS ---
from xai_c.utils import get_model, get_weights_path, get_heatmap_fn
from xai_c.data import get_test_loader

# --- PARAMETERS ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = '/home/ranap/decode_CNN/Dataset/Test/'
MODELS_TO_RUN = ['vgg']
TOP_K_PERCENT = 5  # Threshold: Top 20% salient pixels

# # --- Jaccard Logic ---
# def calculate_jaccard_top_k(map1, map2, top_k_percent=10):
#     """
#     Computes Jaccard Index (IoU) between two heatmaps based on top k% pixels.
#     """
#     m1_flat = map1.flatten()
#     m2_flat = map2.flatten()
    
#     # Number of pixels to keep
#     k = int(len(m1_flat) * (top_k_percent / 100.0))
#     if k == 0: return 0.0

#     # Determine thresholds
#     thresh1 = np.partition(m1_flat, -k)[-k]
#     thresh2 = np.partition(m2_flat, -k)[-k]
    
#     # Binary Masks
#     mask1 = m1_flat >= thresh1
#     mask2 = m2_flat >= thresh2

#     # Intersection over Union
#     intersection = np.logical_and(mask1, mask2).sum()
#     union = np.logical_or(mask1, mask2).sum()
    
#     if union == 0: return 0.0
#     return intersection / union


def calculate_jaccard_top_k(map1, map2, top_k_percent=10):
    """
    Computes Jaccard Index (IoU) between two heatmaps based on strict top k% pixels.
    Fixed to handle binary/discrete maps like LIME without threshold collapse.
    """
    m1_flat = map1.flatten()
    m2_flat = map2.flatten()
    
    k = int(len(m1_flat) * (top_k_percent / 100.0))
    if k == 0: return 0.0

    # Create empty boolean masks
    mask1 = np.zeros_like(m1_flat, dtype=bool)
    mask2 = np.zeros_like(m2_flat, dtype=bool)

    # argsort safely handles ties by forcing exactly 'k' indices to be selected
    idx1 = np.argsort(m1_flat)[-k:]
    idx2 = np.argsort(m2_flat)[-k:]

    mask1[idx1] = True
    mask2[idx2] = True

    # Intersection over Union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0: return 0.0
    return intersection / union

def main():
    print(f"=== Starting Jaccard Analysis using package: xai_c ===")
    print(f"Device: {DEVICE}")
    print(f"Threshold: Top {TOP_K_PERCENT}%")

    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data path {DATA_PATH} not found.")
        return
    # Batch size 1 is required for individual heatmap generation
    loader, _ = get_test_loader(DATA_PATH, batch_size=1)
    
    # Pairs to compare
    pairs = [
        ('LRP', 'PEEK'), ('LRP', 'GradCAM'), ('LRP', 'LIME'),
        ('PEEK', 'GradCAM'), ('PEEK', 'LIME'), ('GradCAM', 'LIME')
    ]

    for model_name in MODELS_TO_RUN:
        print(f"\n" + "="*50)
        print(f"PROCESSING MODEL: {model_name.upper()}")
        print("="*50)

        torch.cuda.empty_cache()
        gc.collect()

        try:
            # 2. Load Model & Methods using Package Utils
            weights = get_weights_path(model_name)
            model = get_model(model_name, weights, DEVICE)

            # Initialize functions
            methods = {
                'LRP': get_heatmap_fn('lrp', model, DEVICE),
                'PEEK': get_heatmap_fn('peek', model, DEVICE),
                'GradCAM': get_heatmap_fn('gradcam', model, DEVICE),
                'LIME': get_heatmap_fn('lime', model, DEVICE)
            }

            print("Running comparisons on all test images...")
            df_data = []

            for i, (img_tensor, _) in enumerate(tqdm(loader)):
                img_tensor = img_tensor.to(DEVICE)
                
                # Generate heatmaps on the fly
                generated_maps = {}
                try:
                    for method_name, fn in methods.items():
                        # fn returns (1, 1, H, W) tensor -> Convert to (H, W) numpy
                        hm = fn(img_tensor).squeeze().detach().cpu().numpy()
                        generated_maps[method_name] = hm
                    
                    # Compute scores
                    for m1, m2 in pairs:
                        score = calculate_jaccard_top_k(
                            generated_maps[m1], generated_maps[m2], TOP_K_PERCENT
                        )
                        df_data.append({
                            'Model': model_name.upper(),
                            'Comparison': f"{m1} vs {m2}",
                            'Jaccard Index': score
                        })
                except Exception as e:
                    # Skip image if any method fails
                    pass
                
                # Periodic Cleanup
                if i % 10 == 0: 
                    torch.cuda.empty_cache()

            # 3. Generate Plot
            if not df_data:
                print("No data collected.")
                continue

            print(f"Plotting results for {model_name.upper()}...")
            df = pd.DataFrame(df_data)
            
            plt.figure(figsize=(10, 6))
            sns.set_style("whitegrid")
            
            # Boxplot consistent with your example style
            ax = sns.boxplot(
                x='Comparison', y='Jaccard Index', data=df, 
                order=[f"{p[0]} vs {p[1]}" for p in pairs],
                palette="Set3", width=0.5
            )
            
            plt.ylim(0, 1.0)
            plt.title(f"Jaccard Similarity (Top {TOP_K_PERCENT}%) - {model_name.upper()}", fontsize=14)
            plt.ylabel("Jaccard Index")
            plt.xlabel("")
            plt.xticks(rotation=45)
            
            if not os.path.exists("Final"): os.makedirs("Final")
            save_path = f"Final/jaccard_similarity_{model_name}.png"
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close() 
            print(f"Saved plot to: {save_path}")

        except Exception as e:
            print(f"Critical Error for {model_name}: {e}")

if __name__ == "__main__":
    main()