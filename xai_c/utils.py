import torch
import os

# Model Loaders
from xai_c.models.definitions import load_cnn, load_vgg16
from xai_c.methods.peek import get_peek_heatmap

def get_weights_path(model_name):
    """Returns default weight paths."""
    if model_name.lower() == "vgg":
        return 'Weights/vgg16_cat_dog.pth'
    elif model_name.lower() == "cnn":
        return 'Weights/cnn_avgpool_Cat_Dog_300.pth'
    raise ValueError(f"Unknown model: {model_name}")

def get_model(model_name, weights_path, device):
    """Loads the specified model."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")
        
    print(f"Loading {model_name.upper()} Model from {weights_path}...")
    
    if model_name.lower() == "cnn":
        return load_cnn(weights_path, device)
    elif model_name.lower() == "vgg":
        return load_vgg16(weights_path, device)
    
    raise ValueError(f"Unknown model: {model_name}")

def get_heatmap_fn(method_name, model, device):
    """
    Factory function that returns a callable: fn(image_tensor) -> heatmap
    """
    method = method_name.lower()
    
    # 1. PEEK (Stateless function)
    if method == 'peek':
        # Wrap to match signature: fn(image) -> heatmap
        # get_peek_heatmap originally takes (model, image)
        return lambda x: get_peek_heatmap(model, x)
        
    # 2. GradCAM (Stateful Wrapper)
    elif method == 'gradcam':
        from xai_c.methods.gradcam import GradCAMWrapper
        gcam = GradCAMWrapper(model)
        return lambda x: gcam(x)
        
    # 3. LRP (Stateful Wrapper)
    elif method == 'lrp':
        from xai_c.methods.lrp import LRPWrapper
        lrp = LRPWrapper(model)
        return lambda x: lrp(x)

    # 4. LIME (Stateful Wrapper)
    elif method == 'lime':
        from xai_c.methods.lime_method import LIMEWrapper
        # Use fewer samples for speed during testing, increase for final results
        lime = LIMEWrapper(model, device, num_samples=100) 
        return lambda x: lime(x)
        
    raise ValueError(f"Unknown method: {method_name}")