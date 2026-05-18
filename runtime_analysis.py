import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import os
import gc
import sys

# Set matplotlib to headless mode
import matplotlib
matplotlib.use('Agg')

# --- SIMPLIFIED IMPORTS FROM PACKAGE ---

from xai_c.utils import get_model, get_weights_path, get_heatmap_fn
from xai_c.data import get_test_loader

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = '/home/ranap/decode_CNN/Dataset/Test/'
MODELS_TO_RUN = ['cnn', 'vgg']
METHODS_TO_RUN = ['LRP', 'PEEK', 'GradCAM', 'LIME']

# Number of images to use for benchmarking. 
# 50 is usually sufficient to get a stable mean/std without waiting hours for LIME.
NUM_SAMPLES = 100

def measure_runtime(model, loader, heatmap_fns, device, num_samples=50):
    """
    Measures runtime for a list of methods on a specific model.
    """
    results = []
    
    # 1. Warmup GPU (Run one dummy pass if using CUDA)
    if device.type == 'cuda':
        dummy_img = torch.zeros((1, 3, 224, 224)).to(device)
        model(dummy_img)
        torch.cuda.synchronize()

    print(f"   Benchmarking on {num_samples} images...")
    
    # Iterate through the loader
    for i, (image, _) in enumerate(tqdm(loader, total=num_samples, desc="Timing")):
        if i >= num_samples:
            break
            
        image = image.to(device)
        
        for method_name, fn in heatmap_fns.items():
            try:
                # Force synchronization before start
                if device.type == 'cuda': torch.cuda.synchronize()
                
                start_time = time.time()
                
                # --- EXECUTION ---
                _ = fn(image) # Generate heatmap
                # -----------------
                
                # Force synchronization after end
                if device.type == 'cuda': torch.cuda.synchronize()
                end_time = time.time()
                
                duration = end_time - start_time
                results.append({
                    'Method': method_name,
                    'Time (s)': duration
                })
                
            except Exception as e:
                print(f"Error measuring {method_name}: {e}")
                
    return results

def main():
    print(f"=== Starting Runtime Analysis ===")
    print(f"Device: {DEVICE}")
    print(f"Samples per model: {NUM_SAMPLES}")

    if not os.path.exists(DATA_PATH):
        print(f"Error: Data path {DATA_PATH} not found.")
        return
    
    # Batch size 1 required for individual XAI generation
    loader, _ = get_test_loader(DATA_PATH, batch_size=1)
    
    all_data = []

    for model_name in MODELS_TO_RUN:
        print(f"\n" + "="*50)
        print(f"BENCHMARKING MODEL: {model_name.upper()}")
        print("="*50)

        torch.cuda.empty_cache()
        gc.collect()

        try:
            # Load Model
            weights = get_weights_path(model_name)
            model = get_model(model_name, weights, DEVICE)

            # Initialize Methods (Consistent Parameters)
            # PEEK=Layer0, LIME=500 samples, GradCAM=Prediction
            heatmap_fns = {
                method: get_heatmap_fn(method.lower(), model, DEVICE) 
                for method in METHODS_TO_RUN
            }

            # Measure
            model_results = measure_runtime(model, loader, heatmap_fns, DEVICE, NUM_SAMPLES)
            
            # Add Model Label
            for res in model_results:
                res['Model'] = model_name.upper()
                all_data.append(res)

        except Exception as e:
            print(f"Critical error benchmarking {model_name}: {e}")

    # --- Plotting ---
    if not all_data:
        print("No data collected.")
        return

    print("\nGenerating Runtime Plot...")
    df = pd.DataFrame(all_data)

    # Calculate Summary Stats for Printing
    summary = df.groupby(['Model', 'Method'])['Time (s)'].agg(['mean', 'std'])
    print("\n--- Runtime Summary (Seconds) ---")
    print(summary)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.4)

    # Barplot with Standard Deviation (ci='sd')
    # Use log scale on Y axis if LIME is huge compared to others
    # But user requested standard barplot style.
    ax = sns.barplot(
        x='Method', 
        y='Time (s)', 
        hue='Model', 
        data=df, 
        palette="muted", 
        capsize=0.1, 
        edgecolor=".2",
        errorbar='sd' # Show Standard Deviation
    )

    plt.title("Average Runtime per Image (Inference + Explanation)", fontsize=16, fontweight='bold')
    plt.ylabel("Time (seconds)")
    plt.xlabel("")
    plt.legend(title="Model")
    
    # Check if we need log scale (if LIME is > 10x slower than others)
    plt.yscale('log') # Uncomment if LIME makes other bars invisible

    if not os.path.exists("Final"): os.makedirs("Final")
    save_path = "Final/runtime_analysis.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")

if __name__ == "__main__":
    main()