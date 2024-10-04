import pytest
from unittest.mock import patch, MagicMock
from base.dataset import GenericDataLoader, Transformer, GenericDataset
from torch.utils.data import DataLoader
from dataset import InputLoader, TargetLoader, UnetDataset, UNetDataModule
import torch
import numpy as np

img_dir = r'D:\CZI_Scope\code\preprocess\training_set\imgs'
mask_dir = r'D:\CZI_Scope\code\preprocess\training_set\masks'


@pytest.fixture
def mock_tiff_memmap():
    with patch('tifffile.memmap') as mock:
        yield mock


@pytest.fixture
def mock_np_load():
    with patch('numpy.load') as mock:
        yield mock

def mock_loader():
    loader = MagicMock()
    loader.files = ['file_0', 'file_1']
    loader.split.return_value = (np.zeros((32, 32)), 'file_0')
    return loader

def test_input_loader_init(mock_tiff_memmap):
    mock_tiff_memmap.return_value = np.zeros((100, 100))
    loader = InputLoader(directory=img_dir, patch_size=32, step_size=16)
    assert len(loader.files) > 0
    assert loader.img_height == 100
    assert loader.img_width == 100


def test_target_loader_init(mock_np_load):
    mock_np_load.return_value = {'mask': np.zeros((100, 100))}
    loader = TargetLoader(directory=mask_dir, patch_size=32, step_size=16)
    assert len(loader.files) > 0
    assert loader.img_height == 100
    assert loader.img_width == 100


def test_input_loader_len(mock_tiff_memmap):
    mock_tiff_memmap.return_value = np.zeros((100, 100))
    loader = InputLoader(directory=img_dir, patch_size=32, step_size=16)
    # 36 patches for 100x100 image with 32x32 patches and 16 step size
    assert len(loader) == len(loader.files) * 36


def test_target_loader_len(mock_np_load):
    mock_np_load.return_value = {'mask': np.zeros((100, 100))}
    loader = TargetLoader(directory=mask_dir, patch_size=32, step_size=16)
    assert len(loader) == len(loader.files) * 36


def test_input_loader_getitem(mock_tiff_memmap):
    mock_tiff_memmap.return_value = np.zeros((100, 100))
    loader = InputLoader(directory=img_dir, patch_size=32, step_size=16)
    patch = loader[0]
    assert patch.shape == (32, 32)


def test_target_loader_getitem(mock_np_load):
    mock_np_load.return_value = {'mask': np.zeros((100, 100))}
    loader = TargetLoader(directory=mask_dir, patch_size=32, step_size=16)
    patch = loader[0]
    assert patch.shape == (32, 32)


def test_unet_dataset_unpack():
    input_loader = mock_loader()
    target_loader = mock_loader()
    # add split_data method to input_loader and target_loader 
    input_loader.split_data.return_value = (np.zeros((32, 32)), 'file_0')
    target_loader.split_data.return_value = (np.zeros((32, 32)), 'file_0')
    dataset = UnetDataset(input_loader, target_loader)
    inputs = torch.zeros((32, 32))
    targets = torch.zeros((32, 32))
    filenames = 'file_0'
    unpacked = dataset.unpack((inputs, filenames), targets)
    assert unpacked == (inputs, targets, filenames)


def test_unet_data_module_train_dataloader():
    input_loader = mock_loader()
    target_loader = mock_loader()
    data_module = UNetDataModule(input_loader, target_loader)
    data_module.setup('fit')
    dataloader = data_module.train_dataloader()
    assert isinstance(dataloader, DataLoader)


def test_unet_data_module_val_dataloader():
    input_loader = mock_loader()
    target_loader = mock_loader()
    data_module = UNetDataModule(input_loader, target_loader)
    data_module.setup('fit')
    dataloader = data_module.val_dataloader()
    assert isinstance(dataloader, DataLoader)


def test_unet_data_module_test_dataloader():
    input_loader = mock_loader()
    target_loader = mock_loader()
    data_module = UNetDataModule(
        input_loader, target_loader, test_loaders=(input_loader, target_loader))
    data_module.setup('test')
    dataloader = data_module.test_dataloader()
    assert isinstance(dataloader, DataLoader)


def test_unet_data_module_state_dict():
    input_loader = mock_loader()
    target_loader = mock_loader()
    data_module = UNetDataModule(input_loader, target_loader)
    state_dict = data_module.state_dict()
    assert 'input_loader' in state_dict
    assert 'target_loader' in state_dict


if __name__ == '__main__':
    pytest.main(["unet_2/test_dataset.py"])
