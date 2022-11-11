import gym
import gym.envs.box2d.lunar_lander as lunar_lander
import numpy as np

from core import EnvInfo, Env
from core import ContinuousNormal, ContinuousBeta, Boolean, Categorical


X = 'x'
Y = 'y'
VX = 'vx'
VY = 'vy'
ANG = 'angle'
V_ANG = 'v_angle'
LEG1 = 'leg_landed_1'
LEG2 = 'leg_landed_2'
ENG = 'engine'
CRASH = 'crash'
REST = 'rest'
FUEL_COST = 'fuel_cost'

NEXT = {name: Env.name_next(name) for name in
        (X, Y, VX, VY, ANG, V_ANG, LEG1, LEG2)}


class LunarLander(Env):
    FUEL_COSTS = np.array([0., 0.03, 0.3, 0.03])

    def __init__(self, render=False):
        if render:
            self.__core = lunar_lander.LunarLander(render_mode='human')
        else:
            self.__core = lunar_lander.LunarLander()

        env_info = EnvInfo()
        env_info.state(X, ContinuousNormal(scale=None))
        env_info.state(Y, ContinuousNormal(scale=None))
        env_info.state(VX, ContinuousNormal(scale=None))
        env_info.state(VY, ContinuousNormal(scale=None))
        env_info.state(ANG, ContinuousNormal(scale=None))
        env_info.state(V_ANG, ContinuousNormal(scale=None))
        env_info.state(LEG1, Boolean())
        env_info.state(LEG2, Boolean())
        env_info.action(ENG, Categorical(4))
        env_info.outcome(CRASH, Boolean())
        env_info.outcome(REST, Boolean())
        env_info.outcome(FUEL_COST, ContinuousNormal(scale=None))

        super().__init__(env_info)
    
    def __del__(self):
        self.__core.close()
    
    def __state_variables(self, x):
        return {X: x[0], Y: x[1], VX: x[2], VY: x[3],
                ANG: x[4], V_ANG: x[5], LEG1: bool(x[6]),
                LEG2: bool(x[7])}
    
    def __outcome_variables(self, x, a):
        return {CRASH: self.__core.game_over or abs(x[0]) >= 1.0,
                REST: not self.__core.lander.awake,
                FUEL_COST: self.FUEL_COSTS[a]}

    def init(self, *args, **kargs):
        obs, info = self.__core.reset()
        return self.__state_variables(obs)
    
    def transit(self, actions):
        a = actions[ENG]
        observation, reward, terminated, truncated, info = self.__core.step(a)

        s = self.__state_variables(observation)
        tran = {self.name_next(k): s[k] for k in s.keys()}
        tran.update(self.__outcome_variables(observation, a))

        return tran, info

    def done(self, transition) -> bool:
        return transition[CRASH] or transition[REST]
    
    def reward(self, transition) -> float:
        reward = 0
        tr = transition
        
        r0 = (
            -100 * np.sqrt(tr[X] * tr[X] + tr[Y] * tr[Y])
            - 100 * np.sqrt(tr[VX] * tr[VX] + tr[VY] * tr[VY])
            - 100 * abs(tr[ANG])
            + (10 if tr[LEG1] else 0)
            + (10 if tr[LEG2] else 0)
        )  
        r1 = (
            -100 * np.sqrt(tr[NEXT[X]] * tr[NEXT[X]] + tr[NEXT[Y]] * tr[NEXT[Y]])
            - 100 * np.sqrt(tr[NEXT[VX]] * tr[NEXT[VX]] + tr[NEXT[VY]] * tr[NEXT[VY]])
            - 100 * abs(tr[NEXT[ANG]])
            + (10 if tr[NEXT[LEG1]] else 0)
            + (10 if tr[NEXT[LEG2]] else 0)
        ) 
        
        # And ten points for legs contact, the idea is if you
        # lose contact again after landing, you get negative reward
        
        reward = (r1 - r0) - tr[FUEL_COST]

        if tr[REST]:
            reward += 100
        
        if tr[CRASH]:
            reward -= 100

        return reward

    def random_action(self):
        return {ENG: self.__core.action_space.sample()}
