import torch

from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from load_data import load

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class ConvBlock:
    def __init__(self):
        pass
    def forward():
        pass
