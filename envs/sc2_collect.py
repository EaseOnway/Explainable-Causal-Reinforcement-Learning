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

LOC_COMMAND_CENTERS = [(33, 33), (50, 33)]
LOC_REFINERIES = [(19, 8), (19, 57), (64, 8), (64, 57)]
LOC_DEPOTS = [(i, j) for j in (13, 52) for i in (30, 37, 44, 51)]

MINERAL_FIELD = UnitType(units.Neutral.MineralField)
GAS_FIELD = UnitType(units.Neutral.VespeneGeyser)
REFINERY = UnitType(units.Terran.Refinery)
SUPPLY_DEPOT = UnitType(units.Terran.SupplyDepot)
COMMAND_CENTER = UnitType(units.Terran.CommandCenter)

UNIT_COST = {REFINERY: 75, SCV: 50, COMMAND_CENTER: 400, SUPPLY_DEPOT: 100}

P_CHANGE_JOB = 0.1

STEP_DURATION = 40


def namedarr2dict(a: NamedNumpyArray):
    return {i: int(a[i]) for i in a._index_names[0]}  # type: ignore


class UnitDescriptor:
    def __init__(self, raw: NamedNumpyArray):
        self.__dict = namedarr2dict(raw)
        self.utype = UnitType(raw.unit_type)
    
    @property
    def location(self):
        return self.__dict['x'], self.__dict['y']

    @property
    def built(self):
        return self.build_progress == 100
    
    def __getitem__(self, key: str):
        return int(getattr(self.__raw, key))
    
    def __getattr__(self, name: str):
        try:
            return self.__dict[name]
        except KeyError:
            raise AttributeError(name)


class StateDescriptor:

    def __init__(self, timestep):
        observation = timestep.observation
        self.__timestep = timestep
        self.__observation = timestep.observation
        self.__all_units = [UnitDescriptor(u)
                            for u in observation.feature_units]
        self.__available_actions = observation.available_actions

        self.is_final = bool(timestep.last())
        self.collected_resource = int(timestep.reward)
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
        assert len(self.__units_by_type[COMMAND_CENTER]) >= 1

        self.command_centers = self.__get_command_centers()
        self.depots = self.__get_depots()
        self.refineries = self.__get_refineries()
        self.mineral_fields_left, self.minerals_fields_right = \
            self.__devide_mineral_fields()
        self.command_center_progress = self.__list_progress(self.command_centers)
        self.depot_progress = self.__list_progress(self.depots)
        self.refinery_progress = self.__list_progress(self.refineries)

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

    def __get_command_centers(self):
        ccs = self.__units_by_type[COMMAND_CENTER]
        cc0, cc1 = None, None
        for cc in ccs:
            assert cc.location is not None
            if cc.location[0] < 40:
                cc0 = cc
            else:
                cc1 = cc
        assert cc0 is not None
        return cc0, cc1
    
    def __get_refineries(self):
        out: List[Optional[UnitDescriptor]] = \
            [None for _ in range(len(LOC_REFINERIES))]
        for u in self.units(REFINERY):
            i = (2 * (u.x > 30)) + (u.y > 30)
            out[i] = u
        return tuple(out)
    
    def __get_depots(self):
        out: List[Optional[UnitDescriptor]] = \
            [None for _ in range(len(LOC_DEPOTS))]
        for u in self.units(SUPPLY_DEPOT):
            i = (4 * (u.y > 30)) + (u.x - 26) // 8
            out[i] = u
        return tuple(out)
    
    def __devide_mineral_fields(self):
        left = [u for u in self.__units_by_type[MINERAL_FIELD] if u.x < 30]
        right = [u for u in self.__units_by_type[MINERAL_FIELD] if u.x > 30]
        return left, right

    def __list_progress(self, units: Tuple[Optional[UnitDescriptor], ...]):
        return [0 if u is None else u.build_progress for u in units]


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
    def select_idle_worker(all=True):
        if all:
            return actions.FUNCTIONS.select_idle_worker("select_all")
        else:
            return actions.FUNCTIONS.select_idle_worker("select")

    @staticmethod
    def collect(target: UnitDescriptor):
        assert target.utype == MINERAL_FIELD or target.utype == REFINERY
        return actions.FUNCTIONS.Harvest_Gather_screen("now", target.location)

    @staticmethod
    def train_scv():
        return actions.FUNCTIONS.Train_SCV_quick("now")

    @staticmethod
    def build_refinery(idx: int):
        loc = LOC_REFINERIES[idx]
        return actions.FUNCTIONS.Build_Refinery_screen("now", loc)
    
    @staticmethod
    def build_command_center(idx: int):
        loc = LOC_COMMAND_CENTERS[idx]
        return actions.FUNCTIONS.Build_CommandCenter_screen("now", loc)
    
    @staticmethod
    def build_depot(idx: int):
        loc = LOC_DEPOTS[idx]
        return actions.FUNCTIONS.Build_SupplyDepot_screen("now", loc)


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


def shift_job(s: StateDescriptor):
    
    def salary(u: Optional[UnitDescriptor]):
        if u is None or not u.built:
            return 0
        return np.exp(- u.assigned_harvesters / u.ideal_harvesters)

    select_worker = None
    if s.n_idle_worker > 0:
        select_worker = ActionMaker.select_idle_worker()
    else:
        for u in s.units(SCV):
            if u.order_id_0 == 359 and np.random.rand() < P_CHANGE_JOB:
                select_worker = ActionMaker.select_one(u)
                break
    
    if select_worker is not None:
        cc0, cc1 = s.command_centers
        income_left, mine_left = salary(cc0), random.choice(s.mineral_fields_left)
        income_right, mine_right = salary(cc1), random.choice(s.minerals_fields_right)
        refineries = [r for r in s.refineries if r is not None]
        income_gas = [salary(r) for r in refineries]
        targets = refineries + [mine_left, mine_right]
        incomes = income_gas + [income_left, income_right]
        target = targets[int(np.argmax(incomes))]
        return [select_worker, ActionMaker.collect(target)]
    else:
        return [ActionMaker.noop()]

class Wait(ActionQueue):
    def __init__(self, cond: Callable[[StateDescriptor], bool]):
        super().__init__()
        self.cond = cond

    def arrange(self, s: StateDescriptor) -> Iterable[Any]:
        if self.cond(s):
            self.action_queue.clear()
            return []
        else:
            return shift_job(s)


class Sleep(ActionQueue):
    def __init__(self, max_step=10):
        super().__init__()
        self.max_step = max_step

    def arrange(self, s: StateDescriptor) -> Iterable[Any]:
        if self.n_step >= self.max_step:
            self.action_queue.clear()
            return []
        else:
            return shift_job(s)



N_WORKER = 'n_worker'
DEPOTS = 'depots'
COMMAND_CENTERS = 'command_centers'
REFINERIES = 'refineries'
MONEY = 'money'
TIMESTEP = 'timestep'
BUILD = 'build'
TRAIN_WORKER = 'train_worker'
ILLEGAL_ACTION = 'illegal_action'
COLLECTED_RESOURCE = 'collected resource'

NEXT = {s: Env.name_next(s) for s in (
    N_WORKER, DEPOTS, COMMAND_CENTERS, REFINERIES, MONEY, TIMESTEP,
)}


class MainTask(ActionQueue):
    MAXLEN = 40

    BUILD_UNIT_TYPES = [-1, COMMAND_CENTER, REFINERY, SUPPLY_DEPOT]
    BUILD_UNIT_NAMES = ['none', 'command_center', 'refinery', 'supply_depot']

    def __init__(self, build: int, train_worker: int):
        super().__init__()

        self.utype = self.BUILD_UNIT_TYPES[build]
        self.cost = train_worker * UNIT_COST[SCV]
        if self.utype != -1:
            self.cost += UNIT_COST[self.utype]
        self.train_worker = train_worker
        
        self.illegal = False
    
    def __next_build(self, s: StateDescriptor, utype: UnitType):
        if utype == COMMAND_CENTER:
            lis = s.command_center_progress
            func = ActionMaker.build_command_center
        elif utype == SUPPLY_DEPOT:
            lis = s.depot_progress
            func = ActionMaker.build_depot
        elif utype == REFINERY:
            lis = s.refinery_progress
            func = ActionMaker.build_refinery
        else:
            assert False
        
        try:
            i = lis.index(0)
        except ValueError:
            return None

        return func(i)

    def arrange(self, s: StateDescriptor) -> Iterable[Any]:
        if self.n_step == 0:
            out = [ActionMaker.select_by_type(s, SCV), ActionMaker.noop(),
                   ActionMaker.select_by_type(s, COMMAND_CENTER),
                   ActionMaker.noop()]
            if self.cost > s.money:
                self.illegal = True
            if self.utype != -1:
                b = self.__next_build(s, self.utype)
                if b is None:
                    self.illegal = True
                if not self.illegal:
                    out[1] = b
            
            cc1 = s.command_centers[1]
            ncc = 2 if (cc1 is not None and cc1.built) else 1
            if self.train_worker > min(ncc, s.supply_limit - s.supply_used):
                self.illegal = True
            if self.train_worker > 0 and not self.illegal:
                out[3] = [ActionMaker.train_scv() for _ in range(self.train_worker)]

            return out
        elif self.n_step >= self.MAXLEN:
            self.action_queue.clear()
            return []
        elif len(self.action_queue) == 0:
            return shift_job(s)
        else:
            return []



class SC2Collect(Env):
    @classmethod
    def init_parser(cls, parser):
        pass

    def define(self, args):
        _def = Env.Definition()
        _def.state(N_WORKER, IntegarNormal(scale=None))
        _def.state(DEPOTS, IntegarNormal(8, scale=None))
        _def.state(COMMAND_CENTERS, IntegarNormal(2, scale=None))
        _def.state(REFINERIES, IntegarNormal(4, scale=None))
        _def.state(MONEY, IntegarNormal(scale=None))
        _def.state(TIMESTEP, IntegarNormal(scale=None))
        _def.action(BUILD, NamedCategorical(*MainTask.BUILD_UNIT_NAMES))
        _def.action(TRAIN_WORKER, Categorical(3))
        _def.outcome(ILLEGAL_ACTION, Boolean(scale=None))
        _def.outcome(COLLECTED_RESOURCE, IntegarNormal(scale=None))
        _def.reward("illegal action", [ILLEGAL_ACTION],
                        lambda x: -10 * float(x))
        _def.reward("collected resource", [COLLECTED_RESOURCE],
                    lambda x: 0.1 * x)

        return _def
    
    def launch(self):
        self._pysc2env = self.__make_env()
        self.__need_restart = False

    def __make_env(self):
        return PySC2Env(map_name="CollectMineralsAndGas",
            players=[Agent(Race.terran)],
            agent_interface_format=AgentInterfaceFormat(
                feature_dimensions=Dimensions(screen=84, minimap=64),
                use_feature_units=True),
            game_steps_per_episode=0,  # no time limit
        )

    def get_output_variables(self, next_state: StateDescriptor) -> NamedValues:
        return {
            NEXT[N_WORKER]: next_state.supply_used,
            NEXT[DEPOTS]: next_state.depot_progress,
            NEXT[REFINERIES]: next_state.refinery_progress,
            NEXT[COMMAND_CENTERS]: next_state.command_center_progress,
            NEXT[MONEY]: next_state.money,
            NEXT[TIMESTEP]: self.__i_step,
            ILLEGAL_ACTION: self.__illegal,
            COLLECTED_RESOURCE: self.__sum_collected
        }

    def init_episode(self, *args, **kargs) -> NamedValues:
        if self.__need_restart:
            self._pysc2env.close()
            self._pysc2env = self.__make_env()
            self.__need_restart = False
        
        timestep, = self._pysc2env.reset()
        self.__inner_state = StateDescriptor(timestep)
        self.__i_step = 0
        self.__illegal = False
        self.__sum_collected = 0

        variables = self.get_output_variables(self.__inner_state)
        return {s: variables[s_] for s, s_ in self.nametuples_s}

    def transit(self, actions: NamedValues) -> Tuple[NamedValues, Any]:
        task = MainTask(actions[BUILD], actions[TRAIN_WORKER])
        '''
        task = build_task(actions[BUILD_WORKER],
                          actions[BUILD_MARINE],
                          actions[BUILD_BARRACKS],
                          actions[BUILD_DEPOT])
        '''
        self.__last_task = task
        self.__illegal = False
        s = self.__inner_state
        
        self.__sum_collected = 0
        while True:
            if s.is_final:
                print("Warning: Unexpected Shut Down...")
                break
            try:
                a = task.step(s)
                time_step, = self._pysc2env.step([a])
                self.__i_step += 1
                s = StateDescriptor(time_step)
                self.__sum_collected += s.collected_resource
            except StopIteration:
                break
        
        self.__inner_state = s

        if s.n_units(MINERAL_FIELD) != 16:  # this happens some times...
            self.__need_restart = True
        
        self.__illegal = task.illegal
        return self.get_output_variables(self.__inner_state), \
            {"s": self.__inner_state}
    
    def terminated(self, variables: NamedValues) -> bool:
        return variables[NEXT[TIMESTEP]] >= 800

    def random_action(self) -> NamedValues:
        return {BUILD: random.randint(0, 3), TRAIN_WORKER: random.randint(0, 2)}
