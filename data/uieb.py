import os
import random
from PIL import Image
from typing import Dict
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as ttf
from torchvision.transforms import (RandomCrop, Pad, RandomHorizontalFlip,
                                    RandomVerticalFlip, Resize, ToTensor, Normalize)


class UIEBTrain(Dataset):
    _INPUT_ = 'input'
    _TARGET_ = 'target'

    def __init__(self, folder: str, size: int):
        super(UIEBTrain, self).__init__()
        self._size = size
        self._root = folder
        self._filenames = os.listdir(os.path.join(self._root, self._INPUT_))

    def __len__(self):
        return len(self._filenames)

    def __getitem__(self, item):
        input_img = Image.open(os.path.join(self._root, self._INPUT_, self._filenames[item]))
        target_img = Image.open(os.path.join(self._root, self._TARGET_, self._filenames[item]))
        input_img, target_img = self._aug_data(input_img, target_img)
        return input_img, target_img

    def _aug_data(self, input_img, target_img):
        # padding
        pad_w = self._size - input_img.width if input_img.width < self._size else 0
        pad_h = self._size - input_img.height if input_img.height < self._size else 0
        input_img = Pad(padding=(0, 0, pad_w, pad_h), padding_mode='reflect')(input_img)
        target_img = Pad(padding=(0, 0, pad_w, pad_h), padding_mode='reflect')(target_img)
        # random crop
        i, j, h, w = RandomCrop.get_params(input_img, output_size=(self._size, self._size))
        input_img = ttf.crop(input_img, i, j, h, w)
        target_img = ttf.crop(target_img, i, j, h, w)
        # random flip
        vertical_flip_seed = random.randint(0, 1)
        horizontal_flip_seed = random.randint(0, 1)
        input_img = RandomVerticalFlip(vertical_flip_seed)(input_img)
        input_img = RandomHorizontalFlip(horizontal_flip_seed)(input_img)
        target_img = RandomVerticalFlip(vertical_flip_seed)(target_img)
        target_img = RandomHorizontalFlip(horizontal_flip_seed)(target_img)
        # random rotate
        rand_rotate = random.randint(0, 3)
        input_img = ttf.rotate(input_img, 90 * rand_rotate)
        target_img = ttf.rotate(target_img, 90 * rand_rotate)
        # to tensor
        input_img = ToTensor()(input_img)
        target_img = ToTensor()(target_img)
        # TODO mix up
        # TODO norm
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return input_img, target_img


class UIEBValid(Dataset):
    _INPUT_ = 'input'
    _TARGET_ = 'target'

    def __init__(self, folder: str, size: int):
        super(UIEBValid, self).__init__()
        self._size = size
        self._root = folder
        self._filenames = os.listdir(os.path.join(self._root, self._INPUT_))
        self._transform = Resize((self._size, self._size))

    def __len__(self):
        return len(self._filenames)

    def __getitem__(self, item):
        input_img = Image.open(os.path.join(self._root, self._INPUT_, self._filenames[item]))
        target_img = Image.open(os.path.join(self._root, self._TARGET_, self._filenames[item]))
        input_img, target_img = self._aug_data(input_img, target_img)
        return input_img, target_img

    def _aug_data(self, input_img, target_img):
        # padding
        pad_w = self._size - input_img.width if input_img.width < self._size else 0
        pad_h = self._size - input_img.height if input_img.height < self._size else 0
        input_img = Pad(padding=(0, 0, pad_w, pad_h), padding_mode='reflect')(input_img)
        target_img = Pad(padding=(0, 0, pad_w, pad_h), padding_mode='reflect')(target_img)
        # resize
        input_img = self._transform(input_img)
        target_img = self._transform(target_img)
        # to tensor
        input_img = ToTensor()(input_img)
        target_img = ToTensor()(target_img)
        return input_img, target_img
