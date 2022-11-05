from typing import Any, Dict, List, Optional, Set, Tuple, final
import numpy as np
import abc
import torch
from utils import Shaping
from utils.typings import Shape, ShapeLike
import core.ptype as ptype
import enum


class DType(enum.Enum):
    Numeric = 0  # real number
    Bool = 1  # boolean number
    Index = 2  # integar number

    @property
    def numpy(self):
        return NP_DTYPE_MAP[self]
    
    @property
    def torch(self):
        return TORCH_DTYPE_MAP[self]


NP_DTYPE_MAP = {
    DType.Bool: np.bool8,
    DType.Numeric: np.float64,
    DType.Index: np.int64
}

TORCH_DTYPE_MAP = {
    DType.Bool: torch.bool,
    DType.Numeric: torch.float,
    DType.Index: torch.long
}
    


class VType(abc.ABC):
    """the class describing a variable in the environment. How the
    buffers and neural networks deal with the data relative to this
    variable will be determined by this class.
    """

    
    @abc.abstractproperty
    def shape(self) -> Shape:
        """the shape of the **raw data** of the variable. Used by the
        environment and the buffer."""
        raise NotImplementedError

    @abc.abstractproperty
    def size(self) -> int:
        """the size of each sample of **input data** the variable. Used by
           the neural networks."""
        raise NotImplementedError
    
    @abc.abstractproperty
    def ptype(self) -> ptype.PType:
        """the posterior probability pattern. This is required since the
        neural network infers not deterministic values, but the distribution
        of the variables."""
        raise NotImplementedError
    
    @abc.abstractproperty
    def dtype(self) -> DType:
        """the data type of "raw data" and "label" of the variable. This is
        required by the buffer and network respectively."""
        raise NotImplementedError

    @abc.abstractmethod
    def raw2input(self, batch: torch.Tensor) -> torch.Tensor:
        """encode the raw data into the input of the network.

        Args:
            batch (torch.Tensor): the batched raw data like (batch_size, *shape)

        Returns:
            torch.Tensor: the encoded data like (batchsize, size_input)
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def raw2label(self, batch: torch.Tensor) -> torch.Tensor:
        """encode the raw data into the label of the network.

        Args:
            batch (torch.Tensor): the batched raw data like (batch_size, *shape)

        Returns:
            torch.Tensor: the encoded data like (batchsize, size_label)
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def label2raw(self, batch: torch.Tensor) -> torch.Tensor:
        """encode the label inferred by networks into raw data like
        (batch_size, *shape).
        """

        raise NotImplementedError


class ContinuousBase(VType):
    """Base Class for Continuous Variables"""

    def __init__(self, shape: ShapeLike):
        super().__init__()
        self.__shape = Shaping.as_shape(shape)
        self.__size = Shaping.get_size(shape)
    
    @property
    def shape(self):
        return self.__shape

    @property
    def size(self) -> int:
        """the size of each sample of the variable as torch.Tensor."""
        return self.__size
    
    @property
    def dtype(self):
        return DType.Numeric

    def raw2input(self, batch: torch.Tensor):
        return batch.view(batch.shape[0], -1)
    
    def raw2label(self, batch: torch.Tensor):
        return batch.view(batch.shape[0], -1)
    
    def label2raw(self, batch: torch.Tensor):
        return batch.view(batch.shape[0], *self.shape)


class ContinuousNormal(ContinuousBase):
    """Continuous Variables with normal (gaussian) posterior"""

    def __init__(self, shape: ShapeLike = (),
                 scale: Optional[float] = 1.):
        """
        Args:
            shape (ShapeLike, optional): the shape of the variable. Defaults to
                (), i.e. scalar.
            scale (None or float, optional): the standard deviance of the
                normal distribution. Defaults to 1. If it is None, the scale
                is inferred by the neural networks.
        """
        super().__init__(shape)
        self.__ptype = ptype.Normal(self.size, scale)
    
    @property
    def ptype(self) -> ptype.PType:
        return self.__ptype


class Categorical(VType):
    """Class for Categorical Variables"""

    def __init__(self, k: int):
        """
        Args:
            k (int): the number of categories
        """        

        super().__init__()
        self.__k = k
        self.__ptype = ptype.Categorical(k)
    
    @property
    def shape(self):
        return ()

    @property
    def size(self) -> int:
        return self.__k
    
    @property
    def dtype(self):
        return DType.Index

    def raw2input(self, batch: torch.Tensor):
        return torch.nn.functional.one_hot(batch, self.__k).\
            to(DType.Numeric.torch)
    
    def raw2label(self, batch: torch.Tensor):
        return batch
    
    def label2raw(self, batch: torch.Tensor):
        return batch
    
    @property
    def ptype(self) -> ptype.PType:
        return self.__ptype


class Boolean(VType):
    """Class for Categorical Variables"""

    def __init__(self):
        """
        Args:
            k (int): the number of categories
        """        

        super().__init__()
        self.__ptype = ptype.Categorical(2)

    @property
    def shape(self):
        return ()

    @property
    def size(self) -> int:
        return 1
    
    @property
    def dtype(self):
        return DType.Bool

    def raw2input(self, batch: torch.Tensor):
        return batch.to(DType.Numeric.torch)
    
    def raw2label(self, batch: torch.Tensor):
        return batch.to(DType.Index.torch)
    
    def label2raw(self, batch: torch.Tensor):
        return batch.bool()
    
    @property
    def ptype(self) -> ptype.PType:
        return self.__ptype
