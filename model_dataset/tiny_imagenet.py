
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk


class TinyImageNetDataset(Dataset):
    def __init__(self, img_size=32, train=True, transform=None, **kwargs):
        self.img_size = img_size
        self.train = train
        self.transform = transform

        self.images = []
        self.labels = []
        tiny_imagenet = load_from_disk(f"data/imagenet32-{'train' if train else 'test'}")
        for data in tiny_imagenet:
            self.images.append(data['image'].convert('RGB').resize((self.img_size, self.img_size)))
            self.labels.append(data['label'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
