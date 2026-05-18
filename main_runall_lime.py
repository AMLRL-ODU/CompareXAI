import torch
import os
import sys
import gc
import numpy as np

import matplotlib
matplotlib.use('Agg') # Force headless backend
import matplotlib.pyplot as plt

# --- Import from xai_kit ---
from xai_c.utils import get_model, get_weights_path, get_heatmap_fn
from xai_c.data import get_test_loader
from xai_c.evaluation import evaluate_robustness_batch, evaluate_model_performance
from xai_c.visualization import plot_noise_curves, plot_occlusion_histograms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = '/home/ranap/decode_CNN/Dataset/Test/'
NOISE_LEVELS = np.round(np.arange(0.1, 2.1, 0.1), 1)

# Only LIME, both models
MODELS_TO_RUN = ['cnn', 'vgg']
METHOD = 'lime'

def main():
    print(f"=== Starting LIME Analysis (Heavy Computation) ===")
    print(f"Device: {DEVICE}")

    if not os.path.exists(DATA_PATH): return

    # Load Data
    loader, classes = get_test_loader(DATA_PATH, batch_size=1)

    for model_name in MODELS_TO_RUN:
        print(f"\n" + "#"*60)
        print(f"PROCESSING LIME FOR: {model_name.upper()}")
        print("#"*60)

        torch.cuda.empty_cache()
        gc.collect()

        try:
            weights = get_weights_path(model_name)
            model = get_model(model_name, weights, DEVICE)
            
            # Get LIME Function
            heatmap_fn = get_heatmap_fn(METHOD, model, DEVICE)

            if not os.path.exists("Final"): os.makedirs("Final")
            title_suffix = f"({METHOD.upper()} on {model_name.upper()})"
            noise_file = f"Final/{METHOD}_noise_{model_name}.png"
            occlusion_file = f"Final/{METHOD}_occlusion_{model_name}.png"

            # Noise
            print("   Running Noise Analysis (This will take a long time)...")
            results_by_noise = {}
            for std in NOISE_LEVELS:
                print(f"   Std: {std}...")
                results_by_noise[std] = evaluate_robustness_batch(
                    loader, model, heatmap_fn, DEVICE, mode="noise", noise_std=std
                )
            plot_noise_curves(results_by_noise, NOISE_LEVELS, noise_file, title_suffix)

            # Occlusion
            print("   Running Occlusion Analysis...")
            occlusion_results = evaluate_robustness_batch(
                loader, model, heatmap_fn, DEVICE, mode="occlusion"
            )
            plot_occlusion_histograms(occlusion_results, occlusion_file, title_suffix)

        except Exception as e:
            print(f"!!! Error in LIME for {model_name}: {e}")

    print("\n=== LIME Analysis Complete ===")

if __name__ == "__main__":
    main()