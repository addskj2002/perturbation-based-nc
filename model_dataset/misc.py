
import torchvision
import torch

IMAGE_SIZE = 32
STL_TRANSFORM = torchvision.transforms.Resize(IMAGE_SIZE)

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return torch.stack([self.transform(x), self.transform(x)])