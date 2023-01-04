from .experiment import Experiment
from .model_based_rl import ModelBasedRL
from .model_free_rl import ModelFreeRL
from .fitting import Fitting
from .explain import TrainExplain, TestExplain


Experiment.register('model_based', ModelBasedRL)
Experiment.register('model_free', ModelFreeRL)
Experiment.register('fitting', Fitting)
Experiment.register('train_explain', TrainExplain)
Experiment.register('test_explain', TestExplain)
