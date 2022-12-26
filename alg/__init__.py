from .experiment import Experiment
from .model_based_rl import ModelBasedRL, ModelBasedRLLunarLander
from .model_free_rl import ModelFreeRL
from .fitting import Fitting
from .explain import TrainExplain


Experiment.register('model_based', ModelBasedRL)
Experiment.register('model_based_ll', ModelBasedRLLunarLander)
Experiment.register('model_free', ModelFreeRL)
Experiment.register('fitting', Fitting)
Experiment.register('train_explain', TrainExplain)
