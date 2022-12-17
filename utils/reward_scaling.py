import numpy as np


class RewardScaling:
    def __init__(self, discount: float):
        self.n = 0
        self.mean: float
        self.__s: float
        self.__sum_reward: float = 0.
        self.discount = discount
    
    def state_dict(self):
        if self.n > 0:
            return {'n': self.n, 'mean': self.mean, 's': self.__s,
                    'R': self.__sum_reward}
        else:
            return None

    def load_state_dict(self, dic):
        if dic is None:
            return
        else:
            self.n = dic['n']
            self.mean = dic['mean']
            self.__s = dic['s']
            self.__sum_reward = dic['R']
    
    @property
    def std(self):
        return float(np.sqrt(self.__s / self.n))
    
    def apply(self, reward):
        std = self.std
        if self.n <= 1 or std == 0.:
            return reward
        else:
            return reward / std
    
    def __call__(self, reward: float, done: bool) -> float:
        r = self.__sum_reward * self.discount + reward
        self.n += 1

        if self.n == 1:
            self.mean = r
            self.__s = 0.
        else:
            old_mean = self.mean
            self.mean += (r - old_mean) / self.n
            self.__s += (r - old_mean) * (r - self.mean)
        
        if done:
            self.__sum_reward = 0.
        else:
            self.__sum_reward = r
        
        return self.apply(reward)
