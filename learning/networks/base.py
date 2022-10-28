from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from learning.config import Config, Configured
from core import Batch

import utils.tensorfuncs as T


class BaseNN(nn.Module, Configured):

    def __init__(self, config: Config):
        nn.Module.__init__(self)
        Configured.__init__(self, config)

    def init_parameters(self):
        for p in self.parameters():
            if p.ndim < 2:
                nn.init.normal_(p)
            else:
                nn.init.xavier_normal_(p)
