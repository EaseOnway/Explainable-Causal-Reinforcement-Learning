from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
import os


def _get_log_data(path: str, key='return', smooth= 1, shift=0):
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    items = ea.scalars.Items(key)

    data = []
    for item in reversed(items):
        if len(data) == 0 or data[-1].step > item.step:
            data.append(item)
        else:
            break
    x = [item.step + shift for item in reversed(data)]
    y = [item.value for item in reversed(data)]

    if smooth > 1:
        temp = np.ones(len(y))
        kernel = np.ones(smooth)
        y = np.convolve(kernel, y, 'same') / np.convolve(kernel, temp, 'same')

    return x, y


def plot_data(parent_path: str, key='return', label: Optional[str] = None, color: Optional[str] = None, smooth=1, shift=0):    
    xs = []
    ys = []
    parent = Path(parent_path)
    dirs = os.listdir(parent_path)
    paths = [parent / d for d in dirs if os.path.isdir(parent / d)]
    for path in paths:
        x, y = _get_log_data(str(path / 'log'), key, smooth=smooth, shift=shift)
        xs.extend(x)
        ys.extend(y)
    sns.lineplot(x=xs, y=ys, label=label, color=color)


plt.title('Cart Pole', fontsize=22)
plot_data('experiments/cartpole/model-based-0.1', smooth=3, color='r', shift=400)
# plot_data('experiments/cartpole/model-based-0.2', smooth=3, color='r')
# plot_data('experiments/cartpole/model-based-0.3', smooth=3, color='r')
plot_data('experiments/cartpole/model-based-mlp', smooth=3, color='b', shift=400)
plot_data('experiments/cartpole/model-based-dense', smooth=3, color='orange', shift=400)
plot_data('experiments/cartpole/model-based-recur', smooth=3, color='g', shift=400)
plot_data('experiments/cartpole/model-free', smooth=3, color='m', shift=-800)
# plt.legend(fontsize=14)
plt.xlabel("time steps", fontsize=18)
plt.ylabel('return', fontsize=18)
plt.xticks([10000, 20000, 30000, 40000], ['10k', '20k', '30k', '40k'], fontsize=16)
plt.yticks(fontsize=16)
plt.xlim(1200, 40000)
plt.tight_layout()
plt.show()
