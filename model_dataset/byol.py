
import torch
from torchvision import models
import torchvision.transforms as transforms

from model_dataset.utils import NetWrapper


def get_model(filename, arch):

    # Build
    if arch == 'resnet18':
        encoder = models.resnet18()
    elif arch == 'resnet50':
        encoder = models.resnet50()
    else:
        raise ValueError

    # Load
    if filename != '':
        encoder.load_state_dict(torch.load(filename, map_location='cpu'))

    encoder_ = NetWrapper(encoder, layer=-2)

    return encoder_

def save_model(model, filename):
    torch.save(model.net.state_dict(), filename)

def get_transform():
    transform = transforms.ToTensor()
    return transform
