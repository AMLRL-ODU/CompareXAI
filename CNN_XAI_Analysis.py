#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai.vision.all import *
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import entropy
from PIL import Image, ImageFilter
from torchvision.transforms.functional import to_tensor
from torch.utils.data import Subset
import random
import torch.nn.functional as F

import matplotlib.cm as cm
import matplotlib.colors as colors

from torch.utils.tensorboard import SummaryWriter
from fastai.callback.tensorboard import TensorBoardCallback
import torchvision
from torchvision import transforms
import cv2
import numpy as np
import os
torch.manual_seed(42)

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

from lime import lime_image

from my_pkg.model import *
from src.peek import *
from src.lrp import *


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


neurons = [8,16,32,64]
in_channels=3
num_classes = 2
depth = 4
train_bs = 64
resize_res = 224
Train = False
weight_path = 'Weights/cnn_avgpool_Cat_Dog.pth'
model = CNN_Avgpool(in_channels=in_channels, num_classes=num_classes, depth=depth, neurons=neurons, input_size=resize_res)
model = model.to(device)
# checkpoint = torch.load(weight_path, map_location=device)
checkpoint = torch.load(weight_path, map_location=device, weights_only=True)
model.features.load_state_dict(checkpoint['features'])
model.classifier.load_state_dict(checkpoint['classifier'])
model.eval()


# In[4]:


def run_peek_and_plot(image_tensor, filename):
    # device = image_tensor.device

    # output = model(image_tensor)
    # probabilities = F.softmax(output, dim=1)
    # predicted_class_index = torch.argmax(probabilities, dim=1)
    # predicted_class_probability = probabilities[0, predicted_class_index]

    # feature_output = image_tensor.clone()
    # for layer in model.features:
    #     feature_output = layer(feature_output)

    feature_output = model.features[0](image_tensor)

    feature_np = feature_output.detach().cpu().numpy()[0]  # [C, H, W]
    feature_np = np.moveaxis(feature_np, 0, -1)            # [H, W, C]

    _, _, h, w = image_tensor.shape

    peek_map = compute_PEEK(feature_np, w, h)

    np.save("../npy_outputs/CNN/PEEK/peek_output_" + filename + ".npy", peek_map)
    print(f"PEEK output saved: {filename}")


# In[5]:


def run_gradcam_and_plot(image_tensor, filename):
    
    img = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    target_layers = [model.features[i*3] for i in range(4)]
    targets = [ClassifierOutputTarget(0), ClassifierOutputTarget(1)]  # Or 1 for dog (adjust if multi-class)
    
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cams = cam(input_tensor=image_tensor, targets=targets)
        cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True) #Removed transpose here
    

    # --- Save Grad-CAM Output ---
    gradcam_output = grayscale_cams[0, :]  # Get the raw grad-cam as numpy array
    output_file = "../npy_outputs/CNN/GradCAM/gradcam_output_"+filename+".npy"  # Adjust the filename as needed
    np.save(output_file, gradcam_output)  # Save the raw numpy array
    print(f"GradCAM output saved: {filename}")


# In[6]:


def run_lrp_and_plot(image_tensor,filename):
    # device = image_tensor.device  # Get the device of the input tensor
    
    # output = model(image_tensor)
    # probabilities = F.softmax(output, dim=1)
    # predicted_class_index = torch.argmax(probabilities, dim=1)
    # predicted_class_probability = probabilities[0, predicted_class_index]
    
    lrp_map = lrp_model.forward(image_tensor) 

    # Move tensors to CPU for plotting (if needed)
    lrp_map = lrp_map[-1][0, 0, :, :].detach().cpu().numpy()

    np.save("../npy_outputs/CNN/LRP/lrp_output_" + filename + ".npy", lrp_map)
    print(f"LRP output saved: {filename}")


# In[7]:


def batch_predict(images):
    model.eval()
    batch = torch.stack([transform(Image.fromarray(img)) for img in images], dim=0)
    batch = batch.to(device)
    with torch.no_grad():
        logits = model(batch)
    return logits.cpu().numpy()
    
def run_lime_and_plot(img,filename):
    
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        np.array(img.resize((224, 224))),
        batch_predict,
        top_labels=5,
        hide_color=0,
        num_samples=1000
        )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    ) 
    np.save("../npy_outputs/CNN/LIME/lime_output_" + filename + ".npy", mask)
    print(f"Lime output saved: {filename}")


# In[8]:


import os

# Set the folder path
folder_path = '../Dataset/Test_merged'

# Supported image file extensions
image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')

# Get list of image file paths
image_paths = [
    os.path.join(folder_path, filename)
    for filename in os.listdir(folder_path)
    if filename.lower().endswith(image_extensions)
]


# In[ ]:


lrp_model = LRPModel(model).to(device)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
for filename in image_paths:
    image =  PILImage.create(filename)
    image_tensor = transform(image).unsqueeze(0).to(device)
    run_lime_and_plot(image,filename[23:])
    run_peek_and_plot(image_tensor,filename[23:])
    run_lrp_and_plot(image_tensor,filename[23:])
    run_gradcam_and_plot(image_tensor,filename[23:])
    # break


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# Load the .npy maps
peek_map = np.load("../npy_outputs/CNN/PEEK/peek_output_Abyssinian_1.jpg.npy")
lrp_map = np.load("../npy_outputs/CNN/LRP/lrp_output_Abyssinian_1.jpg.npy")
gradcam_map = np.load("../npy_outputs/CNN/GradCAM/gradcam_output_Abyssinian_1.jpg.npy")
lime_map = np.load("../npy_outputs/CNN/LIME/lime_output_Abyssinian_1.jpg.npy")

# Set up 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Plot titles and data
titles = ['PEEK', 'LRP', 'LIME', 'GradCAM']
maps = [peek_map, lrp_map, lime_map, gradcam_map]

# Colormap (use 'hot' for a more dramatic heatmap)
cmap = 'afmhot'

# Flatten axes for easy iteration
axes = axes.flatten()

for ax, data, title in zip(axes, maps, titles):
    im = ax.imshow(data, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()


# In[ ]:


import os
import numpy as np
import matplotlib.pyplot as plt

def plot_xai_grid(filename_base, input_dir="../npy_outputs/CNN", output_dir="./XAI_Grids"):
    methods = ['PEEK', 'LRP', 'LIME', 'GradCAM']
    titles = ['PEEK', 'LRP', 'LIME', 'GradCAM']
    cmap = 'afmhot'

    maps = []
    for method in methods:
        # Replace the peek_output_ prefix with the appropriate method prefix
        corrected_name = filename_base.replace("peek_output_", f"{method.lower()}_output_")
        path = os.path.join(input_dir, method, corrected_name)
        if not os.path.exists(path):
            print(f"Skipped {filename_base} — Missing: {path}")
            return
        maps.append(np.load(path))

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"grid_XAI_{filename_base[12:].replace('.npy', '.png')}")

    # Plot 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for ax, data, title in zip(axes, maps, titles):
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

# Batch process all PEEK outputs
peek_dir = "../npy_outputs/CNN/PEEK"
all_files = [f for f in os.listdir(peek_dir) if f.endswith(".npy") and f.startswith("peek_output_")]

for file in all_files:
    plot_xai_grid(file)

