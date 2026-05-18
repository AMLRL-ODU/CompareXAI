import torch

def get_quantile_mask(heatmap, quantile, mask_percent=0.1):
    """
    Generates a boolean mask (1, 1, H, W) from a heatmap.
    """
    # Ensure heatmap is (1, 1, H, W)
    if heatmap.dim() == 2:
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
    elif heatmap.dim() == 3: 
         heatmap = heatmap.sum(dim=0, keepdim=True).unsqueeze(0)

    flat = heatmap.flatten()
    device = heatmap.device

    if quantile == "random":
        num = flat.numel()
        idx = torch.randperm(num, device=device)[:int(mask_percent * num)]
        mask = torch.zeros_like(flat, dtype=torch.bool)
        mask[idx] = True
        return mask.reshape(heatmap.shape)

    # Top: Hide most relevant (highest values)
    if quantile == "top":
        threshold = torch.quantile(flat, 1 - mask_percent)
        return heatmap >= threshold
    
    # Bottom: Hide least relevant (lowest values)
    elif quantile == "bottom":
        threshold = torch.quantile(flat, mask_percent)
        return heatmap <= threshold
    
    else:
        raise ValueError(f"Unknown quantile: {quantile}")

def apply_perturbation(image, mask, mode="noise", std=0.2):
    """
    Applies perturbation to the image based on the mask.
    mask: (1, 1, H, W)
    image: (1, 3, H, W)
    """
    if mode == "noise":
        noise = torch.randn_like(image) * std
        return image + (noise * mask)
    elif mode == "occlusion":
        return image * (~mask) 
    return image