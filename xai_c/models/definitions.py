import torch
import torch.nn as nn
import torchvision.models as models

# --- CNN Definition ---
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 14 * 14, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

# --- Loaders ---

def load_vgg16(weights_path, device):
    """
    Loads VGG16 with a binary classifier (Sigmoid output).
    """
    model = models.vgg16()
    # Modify classifier for binary output as requested
    model.classifier[6] = nn.Sequential(
        nn.Linear(4096, 1),
        nn.Sigmoid()
    )
    
    try:
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        # Fallback for older pytorch versions
        state_dict = torch.load(weights_path, map_location=device)
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def load_cnn(weights_path, device):
    """
    Loads Custom CNN with split state_dict (features/classifier).
    """
    model = CNN()
    
    # Load the dictionary of state dicts
    state_dict = torch.load(weights_path, map_location=device)

    # Load into respective parts
    model.features.load_state_dict(state_dict['features'])
    model.classifier.load_state_dict(state_dict['classifier'])
    
    model.to(device)
    model.eval()
    return model