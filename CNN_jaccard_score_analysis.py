#!/usr/bin/env python
# coding: utf-8

# In[11]:


import os
import numpy as np
import matplotlib.pyplot as plt


# In[12]:


root_dir = "../npy_outputs/CNN"

# Method folders and corresponding file prefixes
methods = {
    "LRP": "lrp_output_",
    "GradCAM": "gradcam_output_",
    "LIME": "lime_output_",
    "PEEK": "peek_output_"
}


# In[13]:


reference_method = "GradCAM"
ref_folder = os.path.join(root_dir, reference_method)
filenames = sorted([f for f in os.listdir(ref_folder) if f.endswith(".npy")])


# In[14]:


def top_10_percent_mask(arr):
    threshold = np.quantile(arr, 0.90)
    return arr >= threshold


# In[15]:


jaccard_scores = {f"{m1} vs {m2}": [] for m1 in methods for m2 in methods if m1 < m2}
high_jaccard_files = {f"{m1} vs {m2}": [] for m1 in methods for m2 in methods if m1 < m2}
zero_jaccard_files = {f"{m1} vs {m2}": [] for m1 in methods for m2 in methods if m1 < m2}


# In[16]:


missing_file_count = 0
comparison_count = 0

for filename in filenames:
    base_id = filename.replace(methods[reference_method], "")
    masks = {}

    for method, prefix in methods.items():
        full_path = os.path.join(root_dir, method, f"{prefix}{base_id}")
        if os.path.exists(full_path):
            try:
                arr = np.load(full_path)
                masks[method] = top_10_percent_mask(arr)
            except Exception as e:
                print(f"Error loading {full_path}: {e}")
        else:
            print(f"[Missing] {method}: {full_path}")
            missing_file_count += 1

    for m1 in methods:
        for m2 in methods:
            if m1 < m2 and m1 in masks and m2 in masks:
                mask1, mask2 = masks[m1], masks[m2]
                intersection = np.logical_and(mask1, mask2).sum()
                union = np.logical_or(mask1, mask2).sum()
                jaccard = intersection / union if union > 0 else 0

                pair_key = f"{m1} vs {m2}"
                jaccard_scores[pair_key].append(jaccard)
                comparison_count += 1

                if jaccard == 0:
                    zero_jaccard_files[pair_key].append((base_id, jaccard))   
                
                elif jaccard >= 0.7:
                    high_jaccard_files[pair_key].append((base_id, jaccard))

print(f"Completed {comparison_count} comparisons")
print(f"Skipped {missing_file_count} missing files")


# In[17]:


print("Mean Jaccard Scores:")
mean_jaccard_scores = {}
for pair, scores in jaccard_scores.items():
    if scores:
        mean_score = np.mean(scores)
        mean_jaccard_scores[pair] = mean_score
        print(f"{pair}: {mean_score:.4f}")
    else:
        print(f"{pair}: No valid comparisons")


# In[18]:


from contextlib import redirect_stdout
with open("CNN_jaccard_log.txt", "w") as f:
    with redirect_stdout(f):
        print("############################################")
        print("#              Jaccard = 0                 #")
        print("############################################")

        for pair, entries in zero_jaccard_files.items():
            if entries:
                print(f"\n{pair} — {len(entries)} files with Jaccard = 0:")
                for base_id, score in entries:
                    print(f"  - {base_id}: {score:.4f}")

        print("\n\n############################################")
        print("#              Jaccard >= 0.7              #")
        print("############################################")

        for pair, entries in high_jaccard_files.items():
            if entries:
                print(f"\n{pair} — {len(entries)} files with Jaccard >= 0.7:")
                for base_id, score in entries:
                    print(f"  - {base_id}: {score:.4f}")


# In[27]:


import os
import numpy as np
import matplotlib.pyplot as plt

# Output directory for histograms
output_dir = "../Figures/CNN_jaccard_Individual_histograms"
os.makedirs(output_dir, exist_ok=True)

# Bin settings
bins = np.linspace(0, 1.0, 51)  # 50 bins between 0 and 1.0

# Plot each pair's Jaccard histogram
for pair, scores in jaccard_scores.items():
    if scores:
        mean_score = np.mean(scores)  # calculate mean

        plt.figure(figsize=(8, 5))
        plt.hist(scores, bins=bins, alpha=0.7, color='teal', edgecolor='black')
        plt.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f"Mean = {mean_score:.2f}")

        plt.xlabel("Jaccard Score")
        plt.ylabel("Frequency")
        plt.title(f"Jaccard Score Distribution: {pair}")
        plt.xlim(0, 1.0)
        plt.ylim(0, 350)
        plt.xticks(np.arange(0, 1.01, 0.1))
        plt.yticks(np.arange(0, 351, 50))
        plt.legend()
        plt.tight_layout()

        filepath = os.path.join(output_dir, f"{pair.replace(' ', '_')}_hist.png")
        plt.savefig(filepath)
        plt.close()

        print(f"Saved: {filepath}")
    else:
        print(f"No valid scores for: {pair}")


# In[28]:


import os
import matplotlib.pyplot as plt
from PIL import Image

# Folder containing the histograms
input_dir = "../Figures/CNN_jaccard_Individual_histograms"
output_pdf = "../Figures/CNN_histograms_merged.pdf"

# Get all PNG files
image_files = sorted([
    os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".png")
])

# Layout config
plots_per_row = 3
num_images = len(image_files)
num_rows = (num_images + plots_per_row - 1) // plots_per_row  # ceiling division

# Create figure and axes
fig, axs = plt.subplots(num_rows, plots_per_row, figsize=(15, 5 * num_rows))
axs = axs.flatten() if num_rows > 1 else [axs]

# Populate subplots with images
for i, ax in enumerate(axs):
    if i < num_images:
        img = Image.open(image_files[i])
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(os.path.basename(image_files[i]).replace("_hist.png", "").replace("_", " "))
    else:
        ax.axis('off')  # blank if no image

# Add the title line
fig.suptitle("Comparison for the Custom CNN Model", fontsize=18, fontweight='bold', y=0.98)

# Save the PDF
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit title
fig.savefig(output_pdf, format='pdf')
plt.close()

print(f"PDF saved as: {output_pdf}")


# In[ ]:




