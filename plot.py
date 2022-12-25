from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator


def _get_log_data(path: str, key='return', smooth= 1):
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    items = ea.scalars.Items(key)
    x = [item.step for item in items]
    y = [item.value for item in items]

    if smooth > 1:
        temp = np.ones(len(y))
        kernel = np.ones(smooth)
        y = np.convolve(kernel, y, 'same') / np.convolve(kernel, temp, 'same')

    return x, y


def plot_data(*paths: str, key='return', label: Optional[str] = None, color: Optional[str] = None, smooth=1):
    xs = []
    ys = []
    for path in paths:
        x, y = _get_log_data(str(Path(path) / 'log'), key, smooth=smooth)
        xs.extend(x)
        ys.extend(y)
    sns.lineplot(x=xs, y=ys, label=label, color=color)


plot_data('experiments/buildmarine/model-based/test', label='causal', color='red', smooth=5)
plt.xlabel("time steps")
plt.ylabel('return')
plt.show()
