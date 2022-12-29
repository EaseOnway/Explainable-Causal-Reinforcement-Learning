from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
import os


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


def plot_data(parent_path: str, key='return', label: Optional[str] = None, color: Optional[str] = None, smooth=1):    
    xs = []
    ys = []
    parent = Path(parent_path)
    dirs = os.listdir(parent_path)
    paths = [parent / d for d in dirs if os.path.isdir(parent / d)]
    for path in paths:
        x, y = _get_log_data(str(path / 'log'), key, smooth=smooth)
        xs.extend(x)
        ys.extend(y)
    sns.lineplot(x=xs, y=ys, label=label, color=color)


plt.title('Cart Pole', fontsize=18)
plot_data('experiments/cartpole/model-based', label='ours', smooth=3, color='r')
plot_data('experiments/cartpole/model-based-mlp', label='mlp', smooth=3, color='b')
plot_data('experiments/cartpole/model-based-dense', label='dense', smooth=3, color='orange')
plot_data('experiments/cartpole/model-based-recur', label='recurrent', smooth=3, color='g')
plot_data('experiments/cartpole/model-free', label='model-free', smooth=3, color='m')
plt.legend(fontsize=12)
plt.xlabel("time steps", fontsize=14)
plt.ylabel('return', fontsize=14)
plt.xticks([10000, 20000, 30000, 40000], ['10k', '20k', '30k', '40k'], fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(0, 40000)
plt.show()
