import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM

class GradCAMWrapper:
    def __init__(self, model):
        self.model = model
        self.target_layers = self._find_last_conv_layer()
        
        # Initialize the GradCAM object once
        # Note: We assume model is already on the correct DEVICE
        self.cam = GradCAM(model=self.model, target_layers=self.target_layers)

    def _find_last_conv_layer(self):
        """
        Automatically finds the last Conv2d layer in model.features.
        Works for both your VGG and CNN definitions.
        """
        last_conv = None
        # Iterate through features to find the last Conv2d
        if hasattr(self.model, 'features'):
            for layer in self.model.features:
                if isinstance(layer, nn.Conv2d):
                    last_conv = layer
        
        if last_conv is None:
            raise ValueError("Could not automatically locate a Conv2d layer in model.features.")
            
        return [last_conv]

    def __call__(self, image_tensor):
        """
        Generates GradCAM heatmap.
        Args:
            image_tensor: (1, 3, H, W)
        Returns:
            heatmap: (1, 1, H, W) tensor
        """
        # targets=None means it calculates CAM for the highest scoring class
        # This matches the behavior of "Explaining the prediction"
        grayscale_cam = self.cam(input_tensor=image_tensor, targets=None)
        
        # Output from library is numpy (Batch, H, W) -> Convert to Tensor (1, 1, H, W)
        heatmap = torch.from_numpy(grayscale_cam).float().to(image_tensor.device)
        return heatmap.unsqueeze(1)