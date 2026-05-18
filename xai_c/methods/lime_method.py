import torch
import numpy as np
from lime import lime_image
import torch.nn.functional as F

class LIMEWrapper:
    def __init__(self, model, device, num_samples=100):
        self.model = model
        self.device = device
        self.num_samples = num_samples
        self.explainer = lime_image.LimeImageExplainer()

    def _batch_predict(self, images):
        """
        Adapts the model to accept a list of numpy images.
        """
        self.model.eval()
        
        # Convert list of (H, W, C) numpy images to (N, C, H, W) Tensor
        batch_tensor = torch.stack([
            torch.tensor(img).permute(2, 0, 1) 
            for img in images
        ]).float()

        batch_tensor = batch_tensor.to(self.device)
        
        with torch.no_grad():
            output = self.model(batch_tensor)
            
            # Handle Binary (VGG) vs Multiclass (CNN)
            if output.shape[1] == 1:
                p1 = output
                p0 = 1.0 - p1
                probs = torch.cat([p0, p1], dim=1)
            else:
                probs = F.softmax(output, dim=1)
            
        return probs.cpu().numpy()

    def __call__(self, x):
        """
        Args:
            x: Input image tensor (C, H, W) or (1, C, H, W)
        Returns:
            heatmap: Tensor (1, 1, H, W)
        """
        if x.dim() == 4:
            x = x.squeeze(0)
        
        # Prepare image for LIME
        img_np = x.permute(1, 2, 0).cpu().numpy().astype(np.double)
        
        # Run LIME (Standard Call)
        # We removed 'progress_bar' arg just in case your version doesn't support it,
        # but otherwise we let it print normally.
        explanation = self.explainer.explain_instance(
            image=img_np,
            classifier_fn=self._batch_predict,
            top_labels=1,
            hide_color=0,
            num_samples=self.num_samples
        )

        # Extract Mask
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], 
            positive_only=True, 
            num_features=5, 
            hide_rest=False
        )
        
        # Convert to Tensor (1, 1, H, W)
        heatmap = torch.from_numpy(mask).float().to(self.device)
        return heatmap.unsqueeze(0).unsqueeze(0)