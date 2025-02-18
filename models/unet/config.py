from base.config import BaseConfigHandler
from base.dataset import Transformer
from os import path as os_path
import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Type, List, Union
import torch
import numpy as np
from sklearn.decomposition import PCA

CONFIG_FILE_PATH = 'config.yml'

# path custom tag handler


def path(loader, node):
    seq = loader.construct_sequence(node)
    return os_path.expanduser(os_path.join(*seq))


# register the tag handlerpathjoin
yaml.add_constructor('!path', path)


def ToTensorLong(*args, **kwargs) -> torch.Tensor:
    '''
    Convert a number to a tensor of type long
    '''
    if len(args) == 1:
        x = args[0]
    elif 'image' in kwargs:
        x = kwargs['image']
    elif 'mask' in kwargs:
        x = kwargs['mask']
    else:
        raise ValueError('Invalid argument passed to ToTensorLong')
    x = np.array(x, dtype=np.uint8)
    return torch.tensor(x, dtype=torch.long)


class ComposeTransforms:
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, image=None, mask=None):
        for transform in self.transforms:
            # recursively apply the transforms
            res = transform(image=image, mask=mask)
            if isinstance(res, tuple):
                image, mask = res
            elif isinstance(res, dict):
                image = res['image']
                mask = res['mask']
            else:
               return res 

        return {'image': image, 'mask': mask}


def get_train_transform():
    return {
        "input": ComposeTransforms(
                                   A.Compose([
                                       A.ToFloat(always_apply=True),
                                       #A.Resize(128, 128),
                                       #A.LongestMaxSize(512),
                                       ToTensorV2(),
                                   ])),
        "target": ComposeTransforms(
            #A.Resize(128, 128),
            ToTensorLong,
        )
    }


def get_val_transform():
    return {
        "input": ComposeTransforms(SuperPixelTransform(),
                                   A.Compose([
                                       A.ToFloat(always_apply=True),
                                       #A.Resize(128, 128),
                                       #A.LongestMaxSize(512),
                                       ToTensorV2(),
                                   ])),
        "target": ComposeTransforms(
            #A.Resize(128, 128),
            ToTensorLong,
        )
    }


class SuperPixelTransform():
    def __init__(self, n_segments=100, p=4, r=20, bins=32):
        self.n_segments = n_segments
        self.p = p
        self.r = r
        self.n_bins = bins # default is 32 because testing showed that the median number is between 20 and 30

    def generate_superpixels(self, image):
        from skimage.segmentation import slic
        segments = slic(image, n_segments=self.n_segments, channel_axis=None)
        return segments

    def aggregate_superpixel_features(self, image, superpixels):
        from skimage.feature import local_binary_pattern
        lbp_img = local_binary_pattern(image, self.p, self.r)
        n_bins = self.n_bins
        features = np.zeros(
            (image.shape[0],
             image.shape[1],
             n_bins),
            dtype=np.float32)

        for label in np.unique(superpixels):
            mask = superpixels == label
            lbp_hist, _ = np.histogram(
                lbp_img[mask],
                bins=n_bins, range=(0, n_bins),
                density=True)
            for i in range(n_bins):
                features[mask, i] = lbp_hist[i]

        return features

    def __call__(self, image, mask):
        '''
        Compute the superpixel features and labels for the given image and mask

        Parameters
        ---
        image: np.ndarray
            The input image
        mask: np.ndarray
            The target mask

        Returns
        ---
        Tuple[np.ndarray, np.ndarray]
            The superpixel features and labels
        '''
        superpixels = self.generate_superpixels(image)
        features = self.aggregate_superpixel_features(image, superpixels)
        # return the superpixel features and the mask
        return {'image': features, 'mask': mask}


class UNetTransformer(Transformer):
    def __init__(self):
        super(UNetTransformer, self).__init__(
            get_train_transform(),
            get_val_transform()
        )

    def apply_train(self, input=True, **kwargs):
        if input:
            xformed = self.train_transform['input'](
                image=kwargs.get('image'),
                mask=kwargs.get('mask'))
            return xformed['image'], xformed['mask']
        else:
            return self.train_transform['target'](kwargs.get('mask'))

    def apply_val(self, input=True, **kwargs):
        if input:
            xformed = self.train_transform['input'](
                image=kwargs.get('image'),
                mask=kwargs.get('mask'))
            return xformed['image'], xformed['mask']
        else:
            return self.train_transform['target'](kwargs.get('mask'))

    def __call__(self, inputs, targets) -> List[Type[torch.Tensor]]:
        inputs, targets = self.apply_train(
            input=True, image=inputs, mask=targets)
        targets = self.apply_val(input=False, mask=targets)
        return inputs, targets


class Config(BaseConfigHandler):
    def __init__(self, file_path: str):
        super(Config, self).__init__()
        self.config = yaml.load(open(file_path, 'r'), Loader=yaml.FullLoader)
        pipeline = A.Compose([
            A.ToFloat(always_apply=True),
            ToTensorV2()
        ])

        self.transform = UNetTransformer()

    def __getattr__(self, name: str) -> any:
        return self.config[name]

    def get(self, key, default):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value

    def update(self, key, value):
        self.set(key, value)

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.config, f)

    def load(self, path):
        with open(path, 'r') as f:
            for line in f:
                k, v = line.strip().split('=')
                self.set(k, v)

    def __str__(self):
        return str(self.config)
