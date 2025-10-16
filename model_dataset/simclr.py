
import torch
import torchvision.transforms as transforms

from model_dataset.resnet_big import SupConResNet
from model_dataset.misc import STL_TRANSFORM, TwoCropTransform


def get_model(filename, arch):

    # Build
    model = SupConResNet(name=arch)

    # Load
    ckpt = torch.load(filename, weights_only=False, map_location='cpu')
    state_dict = ckpt['model']
    if torch.cuda.device_count() == 1:
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict

    # match keys
    load_head_keys = sorted([k for k in state_dict if 'head.' in k])
    model_head_keys = sorted([k for k in model.state_dict() if 'head.' in k])
    assert len(load_head_keys) == len(model_head_keys)
    for old_k, new_k in zip(load_head_keys, model_head_keys):
        state_dict[new_k] = state_dict.pop(old_k)

    model.load_state_dict(state_dict)

    # Get Encoder
    encoder = model.encoder

    return encoder

def save_model(model, filename, arch):
    full_model = SupConResNet(name=arch)
    full_model.encoder = model
    torch.save({"model": full_model.state_dict()}, filename)

def get_transform(pretrain):
    if pretrain == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif pretrain == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif pretrain == 'imagenet32':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise NotImplementedError

    normalize = transforms.Normalize(mean=mean, std=std)

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return transform


def get_perturbation_transform(pretrain, downstream):
    transform = get_transform(pretrain)
    if downstream == "stl10":
        transform = transforms.Compose([STL_TRANSFORM, transform])
    return TwoCropTransform(transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        get_transform(pretrain),
    ]))
