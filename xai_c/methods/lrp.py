import torch

# Try importing the external LRPModel class
try:
    from src.lrp import LRPModel
except ImportError:
    LRPModel = None

class LRPWrapper:
    def __init__(self, model):
        if LRPModel is None:
            raise ImportError("Could not import 'LRPModel' from 'src.lrp'. Ensure src/lrp.py exists.")
        
        # Initialize the LRP model wrapper provided in your custom file
        self.lrp_instance = LRPModel(model)

    def __call__(self, image_tensor):
        """
        Standardized call: (1, 3, H, W) -> (1, 1, H, W) heatmap
        """
        # Forward pass through LRP model
        r = self.lrp_instance.forward(image_tensor)
        
        # Take the last layer relevance and sum across channels (dim 1)
        # r is typically a list of tensors, we take the last one: r[-1]
        # Shape: (Batch, Channel, H, W) -> Sum -> (Batch, 1, H, W)
        heatmap = r[-1].sum(dim=1, keepdim=True)
        return heatmap