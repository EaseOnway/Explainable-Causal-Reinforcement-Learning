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
    sns.lineplot(x=xs, y=ys, label=label, color=color, linewidth=2)


if False:
    plt.title('Cartpole', fontsize=22)
    plot_data('experiments/cartpole/model-based-0.1', smooth=5, color='r', shift=400)'
    plot_data('experiments/cartpole/model-based-mlp', smooth=5, color='dodgerblue', shift=400)
    plot_data('experiments/cartpole/model-based-dense', smooth=5, color='forestgreen', shift=400)
    plot_data('experiments/cartpole/model-free', smooth=5, color='gray', shift=-800)
    # plt.legend(fontsize=14)
    plt.xlabel("time steps", fontsize=18)
    plt.ylabel('return', fontsize=18)
    plt.xticks([10000, 20000, 30000, 40000], ['10k', '20k', '30k', '40k'], fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(1200, 40000)
    plt.tight_layout()
    plt.show()
if False:
    plt.title('Build-Marine', fontsize=22)
    plot_data('experiments/buildmarine/model-based', smooth=10, color='r', shift=600-120)
    plot_data('experiments/buildmarine/model-based-mlp', smooth=10, color='dodgerblue', shift=600-120)
    plot_data('experiments/buildmarine/model-based-dense', smooth=10, color='forestgreen', shift=600-120)
    plot_data('experiments/buildmarine/model-free', smooth=10, color='gray', shift=-120)
    # plt.legend(fontsize=14)
    plt.xlabel("time steps", fontsize=18)
    plt.ylabel('return', fontsize=18)
    plt.xticks([8000, 16000, 24000], ['8k', '16k', '24k'], fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(600-120, 24000)
    plt.tight_layout()
    plt.show()
if False:
    plt.title('Lunarlander-Discrete', fontsize=22)
    plot_data('experiments/lunarlander/model-based', smooth=10, color='r', shift=4096-2400)
    plot_data('experiments/lunarlander/model-based-mlp', smooth=10, color='dodgerblue', shift=4096-2400)
    plot_data('experiments/lunarlander/model-based-dense', smooth=10, color='forestgreen', shift=4096-2400)
    plot_data('experiments/lunarlander/model-free', smooth=10, color='gray', shift=-2400)
    # plt.legend(fontsize=14)
    plt.xlabel("time steps", fontsize=18)
    plt.ylabel('return', fontsize=18)
    plt.xticks([120000, 240000, 360000, 480000], ['120k', '240k', '360k', '480k'], fontsize=16)
    plt.yticks([-200, -100, 0, 100], fontsize=16)
    plt.xlim(4096-2400, 480000)
    plt.ylim(-250, 150)
    plt.tight_layout()
    plt.show()
if True:
    plt.title('Lunarlander-Continuous', fontsize=22)
    plot_data('experiments/lunarlander-cont/model-based', smooth=10, color='r', shift=4096-2400)
    plot_data('experiments/lunarlander-cont/model-based-mlp', smooth=10, color='dodgerblue', shift=4096-2400)
    plot_data('experiments/lunarlander-cont/model-based-dense', smooth=10, color='forestgreen', shift=4096-2400)
    plot_data('experiments/lunarlander-cont/model-free', smooth=10, color='gray', shift=-2400)
    # plt.legend(fontsize=14)
    plt.xlabel("time steps", fontsize=18)
    plt.ylabel('return', fontsize=18)
    plt.xticks([120000, 240000, 360000, 480000], ['120k', '240k', '360k', '480k'], fontsize=16)
    plt.yticks([-200, -100, 0, 100], fontsize=16)
    plt.xlim(4096-2400, 480000)
    plt.ylim(-250, 150)
    plt.tight_layout()
    plt.show()
