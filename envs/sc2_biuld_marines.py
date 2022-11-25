from typing import List, Optional, Dict, Tuple, Any, Callable, Iterable, final
from pysc2.env.sc2_env import SC2Env as PySC2Env, Race, Agent
from pysc2.agents.random_agent import RandomAgent
from pysc2.agents.base_agent import BaseAgent
from pysc2.lib.features import AgentInterfaceFormat, Dimensions
from pysc2.lib.named_array import NamedNumpyArray
from pysc2.lib import actions, units
from collections import deque
import numpy as np
import random
import abc

from core import Env, Categorical, Boolean, IntegarNormal
from utils.typings import NamedValues
import utils


UnitType = int
SCV = UnitType(units.Terran.SCV)
COMMAND_CENTER = UnitType(units.Terran.CommandCenter)
MARINE = UnitType(units.Terran.Marine)
MINERAL_FIELD = UnitType(units.Neutral.MineralField)
BARRACKS = UnitType(units.Terran.Barracks)
SUPPLY_DEPOT = UnitType(units.Terran.SupplyDepot)
LOC_BARRACKS = [(30, 10), (30, 50), (40, 10), (40, 50)] + \
    [(50, j) for j in range(10, 70, 20)]
MAXN_BARRACKS = len(LOC_BARRACKS)
IDX_BARRACKS = {loc: i for i, loc in enumerate(LOC_BARRACKS)}
LOC_SUPPLY_DEPOTS = [(i, j) for i in [65, 72, 80] for j in range(12, 61, 7)]
MAXN_SUPPLY_DEPOTS = len(LOC_SUPPLY_DEPOTS)
IDX_SUPPLY_DEPOTS = {loc: i for i, loc in enumerate(LOC_SUPPLY_DEPOTS)}
MAX_SUPPLY = min(15 + 8 * MAXN_SUPPLY_DEPOTS, 200)


class UnitDescriptor:
    def __init__(self, raw: NamedNumpyArray):
        self.utype = UnitType(raw.unit_type)
        self.build_progress = int(raw.build_progress)

        if hasattr(raw, 'x'):
            self.location = int(raw.x), int(raw.y)
        else:
            self.location = None

    @property
    def built(self):
        return self.build_progress == 100


class StateDescriptor:

    def __init__(self, timestep):
        observation = timestep.observation
        self.__timestep = timestep
        self.__observation = timestep.observation
        self.__all_units = [UnitDescriptor(u)
                            for u in observation.feature_units]
        self.__available_actions = observation.available_actions

        self.is_final = bool(timestep.last())
        selected = observation.multi_select \
            if len(observation.multi_select) > 0 \
            else observation.single_select
        self.selected_units = [UnitDescriptor(u) for u in selected]

        self.__units_by_type: Dict[UnitType, List[UnitDescriptor]] = {}
        for unit in self.__all_units:
            try:
                self.__units_by_type[unit.utype].append(unit)
            except KeyError:
                self.__units_by_type[unit.utype] = [unit]
        assert len(self.__units_by_type[COMMAND_CENTER]) == 1

        self.command_center = self.__units_by_type[COMMAND_CENTER][0]

    @property
    def money(self): return int(self.__observation.player.minerals)

    @property
    def supply_used(self): return int(self.__observation.player.food_used)

    @property
    def supply_limit(self): return int(self.__observation.player.food_cap)

    @property
    def n_idle_worker(self): return int(
        self.__observation.player.idle_worker_count)

    def units(self, unit_type: Optional[UnitType] = None):
        if unit_type is None:
            return self.__all_units
        else:
            try:
                return self.__units_by_type[unit_type]
            except KeyError:
                return []

    def n_units(self, unit_type: Optional[UnitType] = None):
        return len(self.units(unit_type))

    def can_do(self, action_id):
        return action_id in self.__available_actions

    def units_not_built(self, unit_type: Optional[UnitType] = None):
        units = self.units(unit_type)
        return [u for u in units if not u.built]


class ActionMaker:
    '''easy-to-use action maker'''

    @staticmethod
    def noop():
        return actions.FUNCTIONS.no_op()

    @staticmethod
    def select_one(unit: UnitDescriptor):
        if unit.location is None:
            raise ValueError("unknown location")
        return actions.FUNCTIONS.select_point("select", unit.location)

    @staticmethod
    def select_by_type(s: StateDescriptor, utype: UnitType):
        units = s.units(utype)
        if len(units) == 0:
            return ActionMaker.noop()
        else:
            return actions.FUNCTIONS.select_point("select_all_type",
                                                  units[0].location)

    @staticmethod
    def select_idle_worker():
        return actions.FUNCTIONS.select_idle_worker("select_all")

    @staticmethod
    def mine(mineral_field: UnitDescriptor):
        assert mineral_field.utype == MINERAL_FIELD
        return actions.FUNCTIONS.Harvest_Gather_screen("now", mineral_field.location)

    @staticmethod
    def train_scv():
        return actions.FUNCTIONS.Train_SCV_quick("now")

    @staticmethod
    def train_marine():
        return actions.FUNCTIONS.Train_Marine_quick("now")

    @staticmethod
    def biuld_next_depot(s: StateDescriptor):
        n = s.n_units(SUPPLY_DEPOT)
        if n < MAXN_SUPPLY_DEPOTS:
            loc = LOC_SUPPLY_DEPOTS[n]
            return actions.FUNCTIONS.Build_SupplyDepot_screen("now", loc)
        else:
            return ActionMaker.noop()

    @staticmethod
    def biuld_next_barracks(s: StateDescriptor):
        n = s.n_units(BARRACKS)
        if n < MAXN_BARRACKS:
            loc = LOC_BARRACKS[n]
            return actions.FUNCTIONS.Build_Barracks_screen("now", loc)
        else:
            return ActionMaker.noop()


class Task:
    def __init__(self):
        self.failed = False
        self.n_step = 0

    def step(self, s: StateDescriptor) -> Any:
        raise StopIteration


class ActionQueue(Task):
    def __init__(self):
        super().__init__()
        self.action_queue = deque()

    def arrange(self, s: StateDescriptor) -> Iterable[Any]:
        return []

    @final
    def step(self, s: StateDescriptor):
        self.action_queue.extend(self.arrange(s))

        while True:
            try:
                a = self.action_queue.popleft()
            except IndexError:
                raise StopIteration
            if not s.can_do(a.function.value):
                self.failed = True
            else:
                self.n_step += 1
                return a


class Wait(ActionQueue):
    def __init__(self, cond: Callable[[StateDescriptor], bool]):
        super().__init__()
        self.cond = cond

    def arrange(self, s: StateDescriptor) -> Iterable[Any]:
        if self.cond(s):
            return []
        else:
            return [ActionMaker.noop()]


class Sleep(ActionQueue):
    def __init__(self, max_step=10):
        super().__init__()
        self.max_step = max_step

    def arrange(self, s: StateDescriptor) -> Iterable[Any]:
        if self.n_step == 0:
            return [ActionMaker.noop() for _ in range(self.max_step)]
        else:
            return []


class TaskQueue(Task):
    def __init__(self, *tasks: Task):
        super().__init__()
        assert len(tasks) > 0, "no arguments"
        self.__tasks = deque(tasks)
    
    def arrange(self, s: StateDescriptor) -> Iterable[Task]:
        return []

    @final
    def step(self, s: StateDescriptor):
        self.__tasks.extend(self.arrange(s))
        
        while True:
            try:
                task = self.__tasks[0]
            except IndexError:
                raise StopIteration
            
            try:
                a = task.step(s)
                break
            except StopIteration:
                if task.failed:
                    self.failed = True
                self.__tasks.popleft()

        self.n_step += 1
        return a
            

class BackToWork(ActionQueue):
    def __init__(self):
        super().__init__()

    def arrange(self, s: StateDescriptor):
        if self.n_step == 0:
            if s.n_idle_worker > 0:
                mineral = random.choice(s.units(MINERAL_FIELD))
                return [ActionMaker.select_idle_worker(),
                        ActionMaker.mine(mineral)]
        return []


class Build(ActionQueue):

    def __init__(self, worker=False, marine=False,
                 barracks=False, depot=False):
        super().__init__()

        self.__worker = worker
        self.__marine = marine
        self.__barracks = barracks
        self.__depot = depot

    def arrange(self, s: StateDescriptor) -> Iterable[Any]:
        todo = []
        if self.__depot or self.__barracks:
            todo.append(ActionMaker.select_by_type(s, SCV))
        if self.__depot:
            todo.append(ActionMaker.biuld_next_depot(s))
        if self.__barracks:
            todo.append(ActionMaker.biuld_next_barracks(s))

        if self.__worker:
            todo.append(ActionMaker.select_one(s.command_center))
            todo.append(ActionMaker.train_scv())

        if self.__marine:
            todo.append(ActionMaker.select_by_type(s, BARRACKS))
            for u in s.units(BARRACKS):
                if u.built:
                    todo.append(ActionMaker.train_marine())

        self.__depot = False
        self.__barracks = False
        self.__worker = False
        self.__marine = False

        return todo


def build_task(worker=False, marine=False,
               barracks=False, depot=False):
    return TaskQueue(
        Build(worker, marine, barracks, depot),
        BackToWork(),
        Sleep(50),
    )


N_WORKER = 'n_worker'
N_BARRACKS = 'n_barracks'
N_MARINE = 'n_marine'
N_SUPPLY_DEPOT = 'n_supply_depot'
MONEY = 'money'
TIMESTEP = 'timestep'
BUILD_WORKER = 'build_worker'
BUILD_MARINE = 'build_marine'
BUILD_BARRACKS = 'build_barracks'
BUILD_DEPOT = 'build_depot'


NEXT = {s: Env.name_next(s) for s in (
    N_WORKER, N_BARRACKS, N_MARINE, MONEY, N_SUPPLY_DEPOT, TIMESTEP,
)}


class SC2BuildMarine(Env):
    def __init__(self):
        _def = Env.Definition()
        _def.state(N_WORKER, IntegarNormal(scale=None))
        _def.state(N_MARINE, IntegarNormal(scale=None))
        _def.state(N_BARRACKS, IntegarNormal(scale=None))
        _def.state(N_SUPPLY_DEPOT, IntegarNormal(scale=None))
        _def.state(MONEY, IntegarNormal(scale=None))
        _def.state(TIMESTEP, IntegarNormal(scale=None))
        _def.action(BUILD_WORKER, Boolean())
        _def.action(BUILD_MARINE, Boolean())
        _def.action(BUILD_DEPOT, Boolean())
        _def.action(BUILD_BARRACKS, Boolean())

        super().__init__(_def)

        self.def_reward("new marines", [N_MARINE, NEXT[N_MARINE]],
                        lambda n, n_: n_ - n)

        self._pysc2env = PySC2Env(map_name="BuildMarines",
            players=[Agent(Race.terran)],
            agent_interface_format=AgentInterfaceFormat(
                feature_dimensions=Dimensions(screen=84, minimap=64),
                use_feature_units=True),
            game_steps_per_episode=0,  # no time limit
        )

    def get_output_variables(self, next_state: StateDescriptor) -> NamedValues:
        return {
            NEXT[N_WORKER]: next_state.n_units(SCV),
            NEXT[N_BARRACKS]: next_state.n_units(BARRACKS),
            NEXT[N_MARINE]: next_state.n_units(MARINE),
            NEXT[N_SUPPLY_DEPOT]: next_state.n_units(SUPPLY_DEPOT),
            NEXT[MONEY]: next_state.money,
            NEXT[TIMESTEP]: self.__i_step,
        }

    def init_episode(self, *args, **kargs) -> NamedValues:
        timestep, = self._pysc2env.reset()
        self.__inner_state = StateDescriptor(timestep)
        self.__i_step = 0
        self.__last_task = Task()

        variables = self.get_output_variables(self.__inner_state)
        return {s: variables[s_] for s, s_ in self.nametuples_s}

    def transit(self, actions: NamedValues) -> Tuple[NamedValues, Any]:
        task = build_task(actions[BUILD_WORKER],
                          actions[BUILD_MARINE],
                          actions[BUILD_BARRACKS],
                          actions[BUILD_DEPOT])
        self.__last_task = task
        s = self.__inner_state
        while True:
            assert not s.is_final
            try:
                a = task.step(s)
                time_step, = self._pysc2env.step([a])
                self.__i_step += 1
                s = StateDescriptor(time_step)
            except StopIteration:
                break
        
        self.__inner_state = s
        return self.get_output_variables(self.__inner_state), \
            {"s": self.__inner_state}

    @property
    def task_failed(self):
        return self.__last_task.failed

    @property
    def task_len(self):
        return self.__last_task.n_step
    
    def terminated(self, variables: NamedValues) -> bool:
        return variables[NEXT[TIMESTEP]] >= 1720

    def random_action(self) -> NamedValues:
        return {BUILD_WORKER: np.random.rand() < 0.5,
                BUILD_MARINE: np.random.rand() < 0.5,
                BUILD_DEPOT: np.random.rand() < 0.5,
                BUILD_BARRACKS: np.random.rand() < 0.5}
