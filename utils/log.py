from typing import Dict, List, Optional, Iterable, Tuple, Union, Any
import utils
import matplotlib.pyplot as plt
import os
import numpy as np


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


_Tag = Union[str, Tuple[str, ...]]


class Log:
    show = plt.show
    subplot = plt.subplot
    figure = plt.figure

    def __init__(self):
        self.__data = []
        self.__sublogs: Dict[str, Log] = {}
        self.__plotargs = {}
        self.__mean = None
        self.__sum = None
    
    def set_plot_arg(self, *tag: str, **kargs):
        self[tag].__plotargs = kargs

    def __call__(self, value):
        self.__data.append(value)
    
    @property
    def data(self):
        return self.__data
    
    def __setitem__(self, tag: _Tag, value):
        if isinstance(tag, str):
            tag = tag,
        log = self
        for t in tag:
            try:
                log = log.__sublogs[t]
            except KeyError:
                log.__sublogs[t] = Log()
                log = log.__sublogs[t]
        log(value)
    
    def __getitem__(self, tag: _Tag):
        try:
            if isinstance(tag, str):
                return self.__sublogs[tag]
            else:
                log = self
                for t in tag:
                    log = log.__sublogs[t]
                return log
        except KeyError:
            return Log()

    def plot(self, x: Optional[Any] = None, **kargs):
        plotargs = utils.Collections.merge_dic(self.__plotargs, kargs)
        if x is None:
            plt.plot(self.__data, **plotargs)
        else:
            plt.plot(x, self.__data, **plotargs)
        return self

    def plots(self, tags_y: Optional[Iterable[_Tag]] = None,
              x: Optional[Any] = None):
        if tags_y is None:
            tags_y = (k for k in self.__sublogs)
        
        for tag_y in tags_y:
            self[tag_y].plot(x, label=tag_y)
            
        plt.legend()

    @property
    def mean(self) -> float:
        if len(self.__data) == 0:
            return np.nan
        if self.__mean is None:
            mean = self.__mean = np.mean(self.__data)
        else:
            mean = self.__mean
        return mean
    
    @property
    def sum(self):
        if self.__sum is None:
            sum = self.__sum = np.sum(self.__data)
        else:
            sum = self.__sum
        return sum
