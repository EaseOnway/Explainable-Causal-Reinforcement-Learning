from typing import Any, Dict, Optional, Set, Tuple, Union


class VarInfo:
    def __init__(self, shape: Union[int, Tuple[int, ...]],
                 dtype: type, default: Optional[Any] = None):
        if isinstance(shape, int):
            shape = (shape,)
        self.shape: Tuple[int, ...] = shape
        self.dtype = dtype
        self.default = default
        self.size = self.__size(shape)
    
    def __size(self, shape):
        s = 1
        for d in shape:
            s *= d
        return s


class TaskInfo:

    def __init__(self):
        self.action_keys: Set[str] = set()
        self.in_state_keys: Set[str] = set()
        self.out_state_keys: Set[str] = set()
        self.in_out_map: Dict[str, str] = {}
        self.outcomes_keys: Set[str] = set()
        self.outcome_weights: Dict[str, float] = {}
        self.varinfos: Dict[str, VarInfo] = {}

    def var(self, name: str, dtype: type,
                 shape: Union[int, Tuple[int, ...]] = (),
                 default: Optional[Any] = None):
        if name in self.varinfos:
            raise ValueError(f"'{name}' already exists")
        self.varinfos[name] = VarInfo(shape, dtype, default)

    def action(self, name: str, dtype: type,
               shape: Union[int, Tuple[int, ...]] = (),
               default: Optional[Any] = None):
        self.var(name, dtype, shape, default)
        self.action_keys.add(name)

    def state(self, in_name: str, out_name: str, dtype: type,
              shape: Union[int, Tuple[int, ...]] = (),
              default: Optional[Any] = None):
        self.var(in_name, dtype, shape, default)
        self.var(out_name, dtype, shape, default)
        self.in_state_keys.add(in_name)
        self.out_state_keys.add(out_name)
        self.in_out_map[in_name] = out_name

    def outcome(self, name: str, weight: float = 1.0,
               default: Optional[Any] = None):
        self.var(name, float, (), default)
        self.outcomes_keys.add(name)
        self.outcome_weights[name] = weight
