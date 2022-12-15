from typing import Dict, Callable
from .lunar_lander import LunarLander
from .sc2_biuld_marines import SC2BuildMarine
from .sc2_collect import SC2Collect
from .taxi import Taxi
from .cartpole import Cartpole
from core import Env


env_makers: Dict[str, Callable[[], Env]] = {
    'lunarlander': (lambda: LunarLander()),
    'lunarlander-c': (lambda: LunarLander(continuous=True)),
    'lunarlander-r': (lambda: LunarLander(render=True)),
    'lunarlander-rc': (lambda: LunarLander(continuous=True, render=True)),
    'buildmarine': (lambda: SC2BuildMarine()),
    'collect': (lambda: SC2Collect()),
    'cartpole': (lambda: Cartpole()),
    'cartpole-r': (lambda: Cartpole(render=True)),
}


def make_env(name: str):
    try:
        maker = env_makers[name]
        return maker()
    except KeyError as e:
        print(f"unsupported environment name {name}")
        print("supported environments are: ", sep='')
        print(', '.join(env_makers.keys()))
        raise e
