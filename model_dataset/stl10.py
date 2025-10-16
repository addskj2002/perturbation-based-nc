
import torchvision


class STL10(torchvision.datasets.STL10):
    def __init__(self, root, train=True, **kwargs):
        split = 'train' if train else 'test'
        super(STL10, self).__init__(root, split=split, **kwargs)
