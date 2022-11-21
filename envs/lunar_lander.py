import gym
import gym.envs.box2d.lunar_lander as lunar_lander
import numpy as np

from core import Env
from core import ContinuousNormal, ContinuousBeta, \
    Boolean, Categorical
from utils.typings import NamedValues


X = 'x'
Y = 'y'
VX = 'vx'
VY = 'vy'
ANG = 'angle'
V_ANG = 'v_angle'
LEG1 = 'leg_landed_1'
LEG2 = 'leg_landed_2'
ENG = 'engine'
MAINENG = 'main_engine'
LATENG = 'leteral_engine'
CRASH = 'crash'
REST = 'rest'
FUEL_COST = 'fuel_cost'

NEXT = {name: Env.name_next(name) for name in
        (X, Y, VX, VY, ANG, V_ANG, LEG1, LEG2)}


class LunarLander(Env):
    DISCRETE_FUEL_COSTS = np.array([0., 0.03, 0.3, 0.03])

    def __init__(self, continuous=False, render=False):
        if render:
            self.__core = lunar_lander.LunarLander(render_mode='human',
                                                   continuous=continuous)
        else:
            self.__core = lunar_lander.LunarLander(continuous=continuous)
        
        self.__continuous = continuous

        env_info = Env.Definition()
        env_info.state(X, ContinuousNormal(scale=None))
        env_info.state(Y, ContinuousNormal(scale=None))
        env_info.state(VX, ContinuousNormal(scale=None))
        env_info.state(VY, ContinuousNormal(scale=None))
        env_info.state(ANG, ContinuousNormal(scale=None))
        env_info.state(V_ANG, ContinuousNormal(scale=None))
        env_info.state(LEG1, Boolean())
        env_info.state(LEG2, Boolean())

        if self.__continuous:
            env_info.action(MAINENG, ContinuousBeta(low=-1., high=1.))
            env_info.action(LATENG, ContinuousBeta(low=-1., high=1.))
        else:
            env_info.action(ENG, Categorical(4))

        env_info.outcome(CRASH, Boolean())
        env_info.outcome(REST, Boolean())
        env_info.outcome(FUEL_COST, ContinuousNormal(scale=None))

        super().__init__(env_info)

        self.def_reward("distance reduction", (X, Y, NEXT[X], NEXT[Y]),
            lambda x, y, x_, y_: 100 * (np.sqrt(x*x + y*y) - np.sqrt(x_*x_ + y_*y_)))
        self.def_reward("velocity reduction", (VX, VY, NEXT[VX], NEXT[VY]),
            lambda x, y, x_, y_: 100 * (np.sqrt(x*x + y*y) - np.sqrt(x_*x_ + y_*y_)))
        self.def_reward("balancing", (ANG, NEXT[ANG]),
            lambda ang, ang_: 100 * (np.abs(ang) - np.abs(ang_)))
        self.def_reward("landed leg_1", (LEG1, NEXT[LEG1]),
            lambda leg, leg_: 10 * (float(leg_) - float(leg)))
        self.def_reward("landed leg_2", (LEG2, NEXT[LEG2]),
            lambda leg, leg_: 10 * (float(leg_) - float(leg)))
        self.def_reward("fuel cost", (FUEL_COST,), lambda x: -x)
        self.def_reward("resting", (REST,), lambda x: 100 if x else 0)
        self.def_reward("crash", (CRASH,), lambda x: -100 if x else 0)

    def __del__(self):
        self.__core.close()
    
    def __state_variables(self, x):
        return {X: x[0], Y: x[1], VX: x[2], VY: x[3],
                ANG: x[4], V_ANG: x[5], LEG1: bool(x[6]),
                LEG2: bool(x[7])}
    
    def __outcome_variables(self, x, a):
        return {CRASH: self.__core.game_over or abs(x[0]) >= 1.0,
                REST: not self.__core.lander.awake,
                FUEL_COST: self.__compute_fuel_cost(a)}
      
    def __compute_fuel_cost(self, a) -> float:
        if self.__continuous:
            a_main = a[0]
            a_lat = a[1]
            m_power = 0.0
            s_power = 0.0
            if (a_main > 0.0):
                m_power = (a_main + 1.0) * 0.5
                assert 0.5 <= m_power <= 1.0
            if np.abs(a_lat) > 0.5:
                s_power = np.abs(a_lat)
                assert 0.5 <= s_power <= 1.0
            return 0.3 * m_power + 0.03 * s_power
        else:
            return self.DISCRETE_FUEL_COSTS[a]

    def init(self, *args, **kargs):
        obs, info = self.__core.reset()
        return self.__state_variables(obs)
    
    def transit(self, actions):
        if self.__continuous:
            a = np.array([actions[MAINENG], actions[LATENG]], dtype=float)
        else:
            a = actions[ENG]

        observation, reward, terminated, truncated, info = \
            self.__core.step(a)  # type: ignore

        s = self.__state_variables(observation)
        tran = {self.name_next(k): s[k] for k in s.keys()}
        tran.update(self.__outcome_variables(observation, a))

        return tran, info

    def terminated(self, transition) -> bool:
        return transition[CRASH] or transition[REST]

    def random_action(self):
        a = self.__core.action_space.sample()
        if self.__continuous:
            return {MAINENG: a[0], LATENG: a[1]}
        else:
            return {ENG: a}
