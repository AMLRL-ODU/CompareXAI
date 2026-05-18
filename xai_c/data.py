import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

def get_transforms():
    """
    Returns the standard transform pipeline for the CNN.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def get_test_loader(data_path, batch_size=1, num_workers=0):
    """
    Creates the DataLoader for the test set.
    """
    data_transform = get_transforms()
    
    test_dataset = datasets.ImageFolder(
        root=data_path,
        transform=data_transform
    )
    
    # Use shuffle=True to get random samples for analysis
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers
    )
    
    return test_loader, test_dataset.classes