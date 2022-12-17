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

from core import Env, Categorical, Binary, IntegarNormal, Boolean, \
    NamedCategorical
from utils.typings import NamedValues
import utils


UnitType = int
SCV = UnitType(units.Terran.SCV)
COMMAND_CENTER = UnitType(units.Terran.CommandCenter)
MARINE = UnitType(units.Terran.Marine)
MINERAL_FIELD = UnitType(units.Neutral.MineralField)
BARRACKS = UnitType(units.Terran.Barracks)
SUPPLY_DEPOT = UnitType(units.Terran.SupplyDepot)
UNIT_COST = {MARINE: 50, SCV: 50, BARRACKS: 150, SUPPLY_DEPOT: 100}
LOC_BARRACKS = [(25, 10), (25, 50), (35, 10), (35, 50)] + \
    [(50, j) for j in (9, 23, 37, 51)]
MAXN_BARRACKS = len(LOC_BARRACKS)
IDX_BARRACKS = {loc: i for i, loc in enumerate(LOC_BARRACKS)}
MAXN_BUILDERS = 4
LOC_SUPPLY_DEPOTS = [(i, j) for i in [60, 67, 74, 81] for j in range(15, 64, 7)]
MAXN_SUPPLY_DEPOTS = len(LOC_SUPPLY_DEPOTS)
IDX_SUPPLY_DEPOTS = {loc: i for i, loc in enumerate(LOC_SUPPLY_DEPOTS)}
MAX_SUPPLY = min(15 + 8 * MAXN_SUPPLY_DEPOTS, 200)


class UnitDescriptor:
    def __init__(self, raw: NamedNumpyArray):
        self.__raw = raw
        self.utype = UnitType(raw.unit_type)
        self.build_progress = int(raw.build_progress)

        if hasattr(raw, 'x'):
            self.location = int(raw.x), int(raw.y)
        else:
            self.location = None

    @property
    def built(self):
        return self.build_progress == 100
    
    def __getitem__(self, key: str):
        return int(getattr(self.__raw, key))


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

    def can_do(self, action):
        if isinstance(action, list):
            for a in action:
                if a.function.value not in self.__available_actions:
                    return False
            return True
        else:
            return action.function.value in self.__available_actions

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
    def build_depot(s: StateDescriptor, idx: int = -1):
        n = s.n_units(SUPPLY_DEPOT)
        idx = n if idx == -1 else idx
        
        if n <= idx < MAXN_SUPPLY_DEPOTS:
            loc = LOC_SUPPLY_DEPOTS[idx]
            return actions.FUNCTIONS.Build_SupplyDepot_screen("now", loc)
        else:
            return ActionMaker.noop()

    @staticmethod
    def build_barracks(s: StateDescriptor, idx: int = -1):
        n = s.n_units(BARRACKS)
        idx = n if idx == -1 else idx

        if n <= idx < MAXN_BARRACKS:
            loc = LOC_BARRACKS[idx]
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
            if not s.can_do(a):
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


'''
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
'''


class Build(ActionQueue):

    def __init__(self, unit_type: UnitType):
        super().__init__()
        self.__type_to_build = unit_type
    
    def __begin(self,  s: StateDescriptor) -> Iterable[Any]:
        todo = []
        money = s.money
        unit_cost = UNIT_COST[self.__type_to_build]
        if self.__type_to_build == SUPPLY_DEPOT:
            todo.append(ActionMaker.select_by_type(s, SCV))
            idx = n = s.n_units(SUPPLY_DEPOT)
            temp = []
            while money >= unit_cost and idx < min(MAXN_SUPPLY_DEPOTS, n + MAXN_BUILDERS):
                temp.append(ActionMaker.build_depot(s, idx))
                money -= unit_cost
                idx += 1
            todo.append(temp)
        elif self.__type_to_build == BARRACKS:
            todo.append(ActionMaker.select_by_type(s, SCV))
            idx = n = s.n_units(BARRACKS)
            temp = []
            while money > unit_cost and idx < min(MAXN_BARRACKS, n + MAXN_BUILDERS):
                temp.append(ActionMaker.build_barracks(s, idx))
                money -= unit_cost
                idx += 1
            todo.append(temp)
        elif self.__type_to_build == SCV:
            todo.append(ActionMaker.select_one(s.command_center))
            todo.append(ActionMaker.train_scv())
        elif self.__type_to_build == MARINE:
            todo.append(ActionMaker.select_by_type(s, BARRACKS))
            temp = []
            for u in s.units(BARRACKS):
                if u.built and money >= unit_cost:
                    temp.append(ActionMaker.train_marine())
                    money -= unit_cost
            todo.append(temp)
        return todo

    def arrange(self, s: StateDescriptor) -> Iterable[Any]:
        if self.n_step == 0:
            return self.__begin(s)
        else:
            return []


class WaitForComplete(ActionQueue):

    def __init__(self, unit_type: UnitType):
        super().__init__()
        self.__type_to_build = unit_type
    
    def __is_being_built(self, s: StateDescriptor):
        if self.__type_to_build in (BARRACKS, SUPPLY_DEPOT):
            for u in s.units(self.__type_to_build):
                if u.build_progress < 100:
                    return True
            return False
        elif self.__type_to_build == SCV:
            return s.command_center['order_progress_0'] > 0
        else:
            for u in s.units(BARRACKS):
                if u['order_progress_0'] > 0:
                    return True
            return False

    def arrange(self, s: StateDescriptor) -> Iterable[Any]:
        if self.__is_being_built(s):
            return [ActionMaker.noop()]
        else:
            return []


N_WORKER = 'n_worker'
N_BARRACKS = 'n_barracks'
N_MARINE = 'n_marine'
N_SUPPLY_DEPOT = 'n_supply_depot'
MONEY = 'money'
TIMESTEP = 'timestep'
# BUILD_WORKER = 'build_worker'
# BUILD_MARINE = 'build_marine'
# BUILD_BARRACKS = 'build_barracks'
# BUILD_DEPOT = 'build_depot'
BUILD = 'build'
INVALID_ACTION = 'invalid_action'


NEXT = {s: Env.name_next(s) for s in (
    N_WORKER, N_BARRACKS, N_MARINE, MONEY, N_SUPPLY_DEPOT, TIMESTEP,
)}


'''
def build_task(worker=False, marine=False,
               barracks=False, depot=False):
    
    return TaskQueue(
        Build(worker, marine, barracks, depot),
        BackToWork(),
        Sleep(50),
    )
'''

def build_task(build_cls: int):
    if build_cls == 0:
        return TaskQueue(BackToWork(), Sleep(30))
    else:
        utype = [SCV, MARINE, BARRACKS, SUPPLY_DEPOT][build_cls - 1]
        return TaskQueue(Build(utype), BackToWork(), Sleep(30), WaitForComplete(utype))

class SC2BuildMarine(Env):
    @classmethod
    def init_parser(cls, parser):
        pass
    
    def define(self, args):
        _def = Env.Definition()
        _def.state(N_WORKER, IntegarNormal(scale=None))
        _def.state(N_MARINE, IntegarNormal(scale=None))
        _def.state(N_BARRACKS, IntegarNormal(scale=None))
        _def.state(N_SUPPLY_DEPOT, IntegarNormal(scale=None))
        _def.state(MONEY, IntegarNormal(scale=None))
        _def.state(TIMESTEP, IntegarNormal(scale=None))
        _def.action(BUILD, NamedCategorical(
            "none", "worker", "marine", "barracks", "supply depot"))
        
        _def.reward("new marines", [N_MARINE, NEXT[N_MARINE]],
                    lambda n, n_: n_ - n)

        return _def

    def launch(self):
        self._pysc2env = self.__make_env()
        self.__need_restart = False

    def __make_env(self):
        return PySC2Env(map_name="BuildMarines",
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
        if self.__need_restart:
            self._pysc2env.close()
            self._pysc2env = self.__make_env()
            self.__need_restart = False
        
        timestep, = self._pysc2env.reset()
        self.__inner_state = StateDescriptor(timestep)
        self.__i_step = 0
        self.__last_task = Task()

        variables = self.get_output_variables(self.__inner_state)
        return {s: variables[s_] for s, s_ in self.nametuples_s}

    def transit(self, actions: NamedValues) -> Tuple[NamedValues, Any]:
        task = build_task(actions[BUILD])
        '''
        task = build_task(actions[BUILD_WORKER],
                          actions[BUILD_MARINE],
                          actions[BUILD_BARRACKS],
                          actions[BUILD_DEPOT])
        '''
        self.__last_task = task
        s = self.__inner_state
        while True:
            if s.is_final:
                print("Warning: Unexpected Shut Down...")
                break
            try:
                a = task.step(s)
                time_step, = self._pysc2env.step([a])
                self.__i_step += 1
                s = StateDescriptor(time_step)
            except StopIteration:
                break
        
        self.__inner_state = s

        if s.n_units(MINERAL_FIELD) != 8:  # this happens some times...
            self.__need_restart = True

        return self.get_output_variables(self.__inner_state), \
            {"s": self.__inner_state}

    @property
    def task_failed(self):
        return self.__last_task.failed

    @property
    def task_len(self):
        return self.__last_task.n_step
    
    def terminated(self, variables: NamedValues) -> bool:
        return variables[NEXT[TIMESTEP]] >= 1600

    def random_action(self) -> NamedValues:
        return {BUILD: random.randint(0, 4)}
