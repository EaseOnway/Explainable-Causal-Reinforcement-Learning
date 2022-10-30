from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Union

import graphviz


class Var:
    def __init__(self, name: Optional[str] = None):
        self._value: Any = None
        self._name = name
    
    @property
    def name(self):
        if self._name is None:
            raise ValueError("unnamed variable")
        else:
            return self._name
    
    def clear(self):
        self._value = None
    
    @property
    def value(self):
        return self._value


class ExoVar(Var):
    def __init__(self, name: Optional[str] = None, default: Any = None):
        super().__init__(name)
        
        self._default = default
    
    @property
    def value(self):
        value = super().value
        if value is not None:
            return value
        elif self._default is not None:
            self._value = self._default
            return self._value
        else:
            raise ValueError("Unassigned exogenous variable")
    
    @value.setter
    def value(self, v):
        self._value = v


class EndoVar(Var):

    def __not_implemented(self, *args):
        raise NotImplementedError("Causal Equation is not Implemented")

    def __init__(self, parents: Sequence[Var],
                 causal_equation: Optional[Callable],
                 name: Optional[str] = None):
        super().__init__(name)

        self.__pa = tuple(parents)
        
        if causal_equation is None:
            self.__caus_eq = self.__not_implemented
        else:
            self.__caus_eq = causal_equation

        # intervention
        self._do = False
        self._do_value: Any = None

        # counterfact
        self._cf_value: Any = None
    
    @property
    def value(self):
        if self._do:
            return self._do_value
        else:
            if self._value is not None:
                return self._value
            else:
                pas = [pa.value for pa in self.parents]
                self._value = self.__caus_eq(*pas)
                return self._value

    @property
    def parents(self):
        return self.__pa
    
    def do(self, value):
        self._do = True
        self._do_value = value
    
    def undo(self):
        self._do = False
        self._do_value = None
    
    def clear(self):
        super().clear()
        self._cf_value = None
    
    @property
    def cfvalue(self) -> Any:
        if self._cf_value is not None:
            return self._cf_value
        else:
            if self._do:
                self._cf_value = self._do_value
                return self._cf_value
            else:
                pas = []
                for pa in self.parents:
                    if isinstance(pa, EndoVar):
                        pas.append(pa.cfvalue)
                    else:
                        pas.append(pa.value)
                self._cf_value = self.__caus_eq(*pas)
                return self._cf_value
    
    @cfvalue.setter
    def cfvalue(self, value):
        self._cf_value = value
     

class StructrualCausalModel:

    def __init__(self):
        self.__vars: Set[Var] = set()
        self.__named_vars: Dict[str, Var] = {}
        self.__endovars: Set[EndoVar] = set()
        self.__exovars: Set[ExoVar] = set()
        self.__counterfacts: Dict[EndoVar, Any] = {}

    def __contains__(self, item):
        if type(item) is str:
            return item in self.__named_vars
        elif isinstance(item, Var):
            return item in self.__vars
        else:
            return False
    
    def __len__(self):
        return len(self.__vars)

    def add(self, var: Var):
        if var in self.__vars:
            return

        if var._name is None:
            index = 1
            while True:
                name = f'VAR[{index}]'
                if name not in self.__named_vars:
                    break
                index += 1
            var._name = name

        name = var.name

        if name in self.__named_vars:
            raise ValueError(f"variable name '{name}' already used")

        self.__vars.add(var)
        self.__named_vars[name] = var
        if isinstance(var, EndoVar):
            self.__endovars.add(var)
            for pa in var.parents:
                if pa not in self.__vars:
                    self.add(pa)
        elif isinstance(var, ExoVar):
            self.__exovars.add(var)
    
    def rm(self, var: Union[Var, str]):
        if isinstance(var, str):
            var = self.__named_vars[var]
        else:
            var = var

        self.__vars.remove(var)
        self.__named_vars.pop(var.name)
        if isinstance(var, EndoVar):
            self.__endovars.remove(var)
        elif isinstance(var, ExoVar):
            self.__exovars.remove(var)
        
        recur_rm: List[Var] = []
        for var_ in self.__endovars:
            if var in var_.parents:
                recur_rm.append(var_)
        for var_ in recur_rm:
            self.rm(var_)

    def __setattr__(self, name, value):
        self.__dict__[name] = value
        if isinstance(value, Var):
            self[name] = value

    def __setitem__(self, name: str, value: Var):
        if value._name is None:
            value._name = name
        if not self.__contains__(value):
            self.add(value)
    
    def __getitem__(self, name: str):
        return self.__named_vars[name]
    
    def clear(self):
        for var in self.variables:
            var.clear()

    @property
    def variables(self):
        return self.__vars

    @property
    def endo_variables(self):
        return self.__endovars

    @property
    def exo_variables(self):
        return self.__exovars
    
    def counterfacts(self, clear=False,
                     valuedic: Dict[EndoVar, Any] = {}, **kargs):
        if clear:
            self.__counterfacts.clear()
        for var, value in valuedic.items():
            if var in self.__endovars:
                self.__counterfacts[var] = value
            else:
                raise ValueError(f"The variable is not in the model")
        for name, value in kargs.items():
            var = self[name]
            if isinstance(var, EndoVar):
                self.__counterfacts[var] = value
            else:
                raise ValueError(f"\"{name}\" is not a endogenous variable")

    def __endo_reset(self):
        for var in self.endo_variables:
            var.clear()
        for var, value in self.__counterfacts.items():
            var.cfvalue = value

    def assign(self, valuedic: Dict[ExoVar, Any] = {}, **kargs):
        for var, value in valuedic.items():
            if var in self.__exovars:
                var.value = value
            else:
                raise ValueError(f"The variable is not in the model")
        for name, value in kargs.items():
            var = self[name]
            if isinstance(var, ExoVar):
                var.value = value
            else:
                raise ValueError(f"\"{name}\" is not a exogenous variable")
        self.__endo_reset()
    
    def valuedic(self):
        return {var.name: var.value for var in self.variables}

    def parentdic(self):
        return {var.name: {pa.name for pa in var.parents}
                for var in self.endo_variables}

    def plot(self, intervened=False, format='png'):
        # biuld graph
        g = graphviz.Digraph(format=format)
        for var in self.__vars:
            g.node(var.name)
        
        for var in self.__endovars:
            j = var.name
            if intervened and var._do:
                i = f"do({j})"
                g.node(i, shape='none')
                g.edge(i, j, color='red')
                
                for pa in var.parents:
                    i = pa.name
                    g.edge(i, j, style='dotted')
            else:
                for pa in var.parents:
                    i = pa.name
                    g.edge(i, j)
        
        return g
