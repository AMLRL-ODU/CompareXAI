import torch
import os
import sys
import gc
import numpy as np

import matplotlib
matplotlib.use('Agg') # Force headless backend
import matplotlib.pyplot as plt

# --- Import from xai_kit ---
# NOTE: If your folder is named 'xai_c', change 'xai_kit' to 'xai_c' below
from xai_c.utils import get_model, get_weights_path, get_heatmap_fn
from xai_c.data import get_test_loader
from xai_c.evaluation import evaluate_robustness_batch, evaluate_model_performance
from xai_c.visualization import plot_noise_curves, plot_occlusion_histograms

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = '/home/ranap/decode_CNN/Dataset/Test/'
NOISE_LEVELS = np.round(np.arange(0.1, 2.1, 0.1), 1)

# Loop Configurations
MODELS_TO_RUN = ['cnn', 'vgg']
METHODS_TO_RUN = ['peek', 'gradcam', 'lrp']

def main():
    print(f"=== Starting Standard XAI Analysis (PEEK, GradCAM, LRP) ===")
    print(f"Device: {DEVICE}")

    # 1. Load Data Once
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data path {DATA_PATH} not found.")
        return
    loader, classes = get_test_loader(DATA_PATH, batch_size=1)
    print(f"Data Loaded. Classes: {classes}")

    # 2. Iterate through Models
    for model_name in MODELS_TO_RUN:
        print(f"\n" + "#"*60)
        print(f"PROCESSING MODEL: {model_name.upper()}")
        print("#"*60)

        # Clean memory before loading new model
        torch.cuda.empty_cache()
        gc.collect()

        try:
            weights_path = get_weights_path(model_name)
            model = get_model(model_name, weights_path, DEVICE)
            
            # Check baseline
            print(f"checking baseline performance for {model_name}...")
            evaluate_model_performance(model, loader, DEVICE, classes)

            # 3. Iterate through Methods
            for method in METHODS_TO_RUN:
                print(f"\n   >>> Method: {method.upper()} on {model_name.upper()}")
                
                try:
                    heatmap_fn = get_heatmap_fn(method, model, DEVICE)
                    
                    if not os.path.exists("Final"): os.makedirs("Final")
                    title_suffix = f"({method.upper()} on {model_name.upper()})"
                    noise_file = f"Final/{method}_noise_{model_name}.png"
                    occlusion_file = f"Final/{method}_occlusion_{model_name}.png"

                    # Noise Analysis
                    print(f"       Running Noise Analysis...")
                    results_by_noise = {}
                    for std in NOISE_LEVELS:
                        print(f"       Std: {std}")
                        results_by_noise[std] = evaluate_robustness_batch(
                            loader, model, heatmap_fn, DEVICE, mode="noise", noise_std=std
                        )
                    plot_noise_curves(results_by_noise, NOISE_LEVELS, noise_file, title_suffix)

                    # Occlusion Analysis
                    print(f"       Running Occlusion Analysis...")
                    occlusion_results = evaluate_robustness_batch(
                        loader, model, heatmap_fn, DEVICE, mode="occlusion"
                    )
                    plot_occlusion_histograms(occlusion_results, occlusion_file, title_suffix)
                    
                    print(f"       Finished {method.upper()}.")
                    
                    # Cleanup after each method
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"       !!! Error with {method}: {e}")

        except Exception as e:
            print(f"!!! Error loading/running {model_name}: {e}")

    print("\n=== All Standard Analysis Complete ===")

if __name__ == "__main__":
    main()