from typing import Any, Dict, List, Optional, Set, Tuple, final, Final
import numpy as np
import abc
import torch
from utils import Shaping
from utils.typings import Shape, ShapeLike
import core.ptype as ptype
import enum


class DType(enum.Enum):
    Real = 0  # real number
    Bool = 1  # boolean number
    Integar = 2  # integar number

    @property
    def numpy(self):
        return NP_DTYPE_MAP[self]
    
    @property
    def torch(self):
        return TORCH_DTYPE_MAP[self]


NP_DTYPE_MAP = {
    DType.Bool: np.bool8,
    DType.Real: np.float64,
    DType.Integar: np.int64
}

TORCH_DTYPE_MAP = {
    DType.Bool: torch.bool,
    DType.Real: torch.float,
    DType.Integar: torch.long
}
    

EPSILON = 1e-5
DECIMAL = 4


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

    def tensor(self, value, device):
        dtype = self.dtype.torch
        if not isinstance(value, torch.Tensor):
            tensor = torch.tensor(value, device=device, dtype=dtype)
        else:
            tensor = value.to(dtype=dtype, device=device)

        if tensor.shape != self.shape:
            raise ValueError("inconsistent shape")
        
        return tensor

    def text(self, value: Any) -> str:
        return str(value)
    
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
        return DType.Real

    def text(self, value: Any) -> str:
        return str(np.round(value, DECIMAL))

    def raw2input(self, batch: torch.Tensor):
        return batch.view(batch.shape[0], -1)
    
    def raw2label(self, batch: torch.Tensor):
        return batch.view(batch.shape[0], -1)
    
    def label2raw(self, batch: torch.Tensor):
        return batch.view(batch.shape[0], *self.shape)


class IntegarNormal(VType):
    """Base Class for Continuous Variables"""

    def __init__(self, shape: ShapeLike = (), scale: Optional[float] = 1.):
        super().__init__()
        self.__shape = Shaping.as_shape(shape)
        self.__size = Shaping.get_size(shape)
        self.__ptype = ptype.Normal(self.size, scale)
    
    @property
    def ptype(self) -> ptype.PType:
        return self.__ptype
    
    @property
    def shape(self):
        return self.__shape

    @property
    def size(self) -> int:
        """the size of each sample of the variable as torch.Tensor."""
        return self.__size
    
    @property
    def dtype(self):
        return DType.Integar

    def raw2input(self, batch: torch.Tensor):
        return batch.view(batch.shape[0], -1).to(
            dtype=DType.Real.torch)
    
    def raw2label(self, batch: torch.Tensor):
        return batch.view(batch.shape[0], -1).to(
            dtype=DType.Real.torch)
    
    def label2raw(self, batch: torch.Tensor):
        return torch.round(batch.view(batch.shape[0], *self.shape)).to(
            dtype=DType.Integar.torch)


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


class TruncatedNormal(ContinuousBase):

    def __init__(self, low: Any, high: Any, shape: ShapeLike = (), 
                 scale: Optional[float] = 1.):

        super().__init__(shape)
        self.__l = torch.tensor(low, dtype=DType.Real.torch)
        self.__h = torch.tensor(high, dtype=DType.Real.torch)
        self.__ptype = ptype.TanhNormal(self.size, scale, (self.__l, self.__h))
    
    @property
    def ptype(self) -> ptype.PType:
        return self.__ptype
    
    def __get_low_high(self, device: torch.device):
        if self.__l.device != device:
            self.__l = self.__l.to(device)
        if self.__h.device != device:
            self.__h = self.__h.to(device)
        return self.__l, self.__h
    
    def label2raw(self, batch: torch.Tensor):
        l, h = self.__get_low_high(batch.device)
        batch = batch.view(batch.shape[0], *self.shape)
        return torch.clamp(batch, l, h)


class ContinuousBeta(ContinuousBase):
    """Continuous Variables with beta posterior"""

    def __init__(self, shape: ShapeLike = (),
                 low: Optional[Any] = None,
                 high: Optional[Any] = None):
        """
        Args:
            shape (ShapeLike, optional): the shape of the variable. Defaults to
                (), i.e. scalar.
            scale (None or float, optional): the standard deviance of the
                normal distribution. Defaults to 1. If it is None, the scale
                is inferred by the neural networks.
        """
        super().__init__(shape)

        if low is None:
            self.__l = None
        else:
            self.__l = torch.tensor(low, dtype=DType.Real.torch)

        if high is None:
            self.__h = None
        else:
            self.__h = torch.tensor(high, dtype=DType.Real.torch)

        self.__ptype = ptype.Beta(self.size)

    @property
    def ptype(self) -> ptype.PType:
        return self.__ptype

    def __get_low_high(self, device: torch.device):
        if self.__l is not None and self.__l.device != device:
            self.__l = self.__l.to(device)
        if self.__h is not None and self.__h.device != device:
            self.__h = self.__h.to(device)

        l = 0. if self.__l is None else self.__l
        h = 1. if self.__h is None else self.__h
        return l, h

    def raw2input(self, batch: torch.Tensor):
                return batch.view(batch.shape[0], -1)
    
    def raw2label(self, batch: torch.Tensor):
        l, h = self.__get_low_high(batch.device)
        x = batch.view(batch.shape[0], -1)
        x = (x - l) / (h - l)
        x[x == 0.] += EPSILON
        x[x == 1.] += EPSILON
        assert (torch.all(torch.logical_and(x<1., x>0.)))
        return x
    
    def label2raw(self, batch: torch.Tensor):
        l, h = self.__get_low_high(batch.device)
        x = batch.view(batch.shape[0], *self.shape)
        x =  l + x * (h - l)
        x[x == l] += EPSILON
        x[x == h] -= EPSILON
        assert (torch.all(torch.logical_and(x<h, x>l)))
        return x


class Categorical(VType):
    """Class for Categorical Variables"""

    def __init__(self, k: int):
        """
        Args:
            k (int): the number of categories
        """        

        super().__init__()
        self.k: Final = k
        self.__ptype = ptype.Categorical(k)

    @property
    def shape(self):
        return ()

    @property
    def size(self) -> int:
        return self.k
    
    @property
    def dtype(self):
        return DType.Integar

    def raw2input(self, batch: torch.Tensor):
        return torch.nn.functional.one_hot(batch, self.k).\
            to(DType.Real.torch)
    
    def raw2label(self, batch: torch.Tensor):
        return batch
    
    def label2raw(self, batch: torch.Tensor):
        return batch
    
    @property
    def ptype(self) -> ptype.PType:
        return self.__ptype


class NamedCategorical(Categorical):
    """Class for Categorical Variables"""

    def __init__(self, *names: str):
        """
        Args:
            k (int): the number of categories
        """        

        super().__init__(len(names))
        self.__names = names

    def text(self, value: Any) -> str:
        return self.__names[int(value)]


class Boolean(IntegarNormal):
    """Class for Categorical Variables"""

    def __init__(self, shape: ShapeLike = (), scale: Optional[float] = 1):
        super().__init__(shape, scale)

    @property
    def dtype(self):
        return DType.Bool

    def raw2input(self, batch: torch.Tensor):
        return torch.where(batch.view(batch.shape[0], -1), 1., -1.).to(
            dtype=DType.Real.torch)
    
    def raw2label(self, batch: torch.Tensor):
        return torch.where(batch.view(batch.shape[0], -1), 1., -1.).to(
            dtype=DType.Real.torch)
    
    def label2raw(self, batch: torch.Tensor):
        return batch.view(batch.shape[0], *self.shape) >= 0


class Binary(Categorical):
    """Class for Categorical Variables"""

    def __init__(self):
        """
        Args:
            k (int): the number of categories
        """        

        super().__init__(2)

    @property
    def shape(self):
        return ()
    
    @property
    def dtype(self):
        return DType.Bool
    
    def raw2input(self, batch: torch.Tensor):
        return super().raw2input(batch.to(DType.Integar.torch))
    
    def raw2label(self, batch: torch.Tensor):
        return batch.to(DType.Integar.torch)
    
    def label2raw(self, batch: torch.Tensor):
        return batch.bool()
