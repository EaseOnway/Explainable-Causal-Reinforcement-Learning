import torch
import torch.nn as nn
from .config import *


class BaseNN(nn.Module):
    Config = NetConfig
    Dims = NetDims
    Ablations = NetAblations

    def __init__(self, config: NetConfig):
        super().__init__()
        self.__config = config
    
    @property
    def config(self):
        return self.__config
    
    @property
    def dims(self):
        return self.__config.dims
    
    @property
    def torchargs(self):
        return self.__config.torchargs

    @property
    def ablations(self):
        return self.__config.ablations