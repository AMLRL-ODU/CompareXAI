import torch
import numpy as np
import cv2
from scipy.special import entr

def compute_PEEK(feature_maps, w, h):
    # feature_maps shape: (H_feat, W_feat, Channels)
    positivized_maps = feature_maps + np.abs(np.min(feature_maps))
    entropy_map = -np.sum(entr(positivized_maps), axis=-1)
    peek_map = cv2.resize(entropy_map, (w, h))
    return peek_map

def get_peek_heatmap(model, image_tensor):
    """
    Extracts PEEK heatmap from a model.
    Works for VGG and CNN provided model.features exists.
    """
    # 1. Forward pass to get features
    # VGG/CNN features output is typically (B, C, H, W)
    with torch.no_grad():
        features = model.features(image_tensor)
    
    # 2. Convert to Numpy for PEEK math
    feat_np = features.detach().cpu().numpy()[0] # (C, H, W)
    feat_np = np.moveaxis(feat_np, 0, -1)        # (H, W, C)
    
    _, _, h, w = image_tensor.shape
    
    # 3. Compute PEEK
    peek_map = compute_PEEK(feat_np, w, h)
    
    # 4. Return as Tensor (1, 1, H, W)
    heatmap = torch.from_numpy(peek_map).float().to(image_tensor.device)
    return heatmap.unsqueeze(0).unsqueeze(0)