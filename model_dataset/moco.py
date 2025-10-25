
import torch
import torchvision.transforms as transforms

from model_dataset.moco_model import ModelMoCo
from model_dataset.utils import NetWrapper


def get_model(filename, arch):

    # Build
    model = ModelMoCo(
        dim=128,
        K=4096,
        m=0.99,
        T=0.1,
        arch=arch,
        bn_splits=8,
        symmetric=False,
    )

    model.load_state_dict(torch.load(filename, map_location='cpu'))

    # Get Encoder
    encoder = model.encoder_q.net

    encoder_ = NetWrapper(encoder, layer=-3)

    return encoder_

def save_model(model, filename):
    wrapper = ModelMoCo(
        dim=128,
        K=4096,
        m=0.99,
        T=0.1,
        arch=arch,
        bn_splits=8,
        symmetric=False,
    )
    wrapper.encoder_q.net = model
    torch.save(wrapper.state_dict(), filename)

def get_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    return transform

