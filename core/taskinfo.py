from typing import Any, Dict, Optional, Sequence, Set, Tuple, Union, TypeVar
import numpy as np
import utils as u


class VarInfo:
    def __init__(self, shape: u.ShapeLike, categorical: bool, dtype: type,
                 default: Optional[Any]):
        self.shape = u.as_shape(shape)
        self.size = u.get_size(shape)
        self.dtype = dtype
        self.default = default
        
        self.categorical = categorical
        if categorical and len(self.shape) == 0:
            raise ValueError("invalid shape for categorical data.")


class TaskInfo:

    def __init__(self):
        self.action_keys: Set[str] = set()
        self.in_state_keys: Set[str] = set()
        self.out_state_keys: Set[str] = set()
        self.in_out_map: Dict[str, str] = {}
        self.outcomes_keys: Set[str] = set()
        self.outcome_weights: Dict[str, float] = {}
        self.varinfos: Dict[str, VarInfo] = {}

    def var(self, name: str, shape: u.ShapeLike = (), categorical=False,
             dtype: type = float, default: Optional[Any] = None):
        if name in self.varinfos:
            raise ValueError(f"'{name}' already exists")
        self.varinfos[name] = VarInfo(shape, categorical, dtype, default)

    def action(self, name: str, shape: u.ShapeLike,
               categorical=True, dtype: type = np.uint, 
               default: Optional[Any] = None):
        self.var(name, shape, categorical, dtype, default)
        self.action_keys.add(name)

    def state(self, in_name: str, out_name: str, categorical=False,
              dtype: type = float, shape: u.ShapeLike = (),
              default: Optional[Any] = None):
        self.var(in_name, shape, categorical, dtype, default)
        self.var(out_name, shape, categorical, dtype, default)
        self.in_state_keys.add(in_name)
        self.out_state_keys.add(out_name)
        self.in_out_map[in_name] = out_name

    def outcome(self, name: str, weight: float = 1.0,
                default: Optional[Any] = None):
        self.var(name, (), False, float, default)
        self.outcomes_keys.add(name)
        self.outcome_weights[name] = weight
