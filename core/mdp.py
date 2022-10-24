from typing import Sequence, Tuple

from .scm import ExoVar, EndoVar, StructrualCausalModel


class CausalMdp(StructrualCausalModel):
    def __init__(self):
        super().__init__()
        self.in_state_vars: Tuple[ExoVar, ...]
        self.out_state_vars: Tuple[EndoVar, ...]
        self.action_vars: Tuple[ExoVar, ...]
        self.reward_vars: Tuple[EndoVar, ...]

    def config(self, state_vars: Sequence[Tuple[ExoVar, EndoVar]],
               action_vars: Sequence[ExoVar],
               reward_vars: Sequence[EndoVar]):
        state_vars_in = []
        state_vars_out = []
        for step_in, step_out in state_vars:
            if step_in not in self.variables:
                self.add(step_in)
            if step_out not in self.variables:
                self.add(step_out)
            state_vars_in.append(step_in)
            state_vars_out.append(step_out)
        self.in_state_vars = tuple(state_vars_in)
        self.out_state_vars = tuple(state_vars_out)

        for a in action_vars:
            if a in self.in_state_vars:
                raise ValueError("argument is already a state variable")
            if a not in self.variables:
                self.add(a)
        self.action_vars = tuple(action_vars)
        
        for r in reward_vars:
            if r not in self.variables:
                self.add(r)
        self.reward_vars = tuple(reward_vars)

    def model(self, state_values: Tuple, action_values: Tuple):
        value_dic = {var: value for var, value in zip(
            self.in_state_vars, state_values)}
        value_dic.update({var: value for var, value in zip(
            self.action_vars, action_values)})
        self.assign(value_dic)

        state_values = tuple(var.value for var in self.out_state_vars)
        rewards = tuple(var.value for var in self.reward_vars)

        return state_values, rewards

    def counterfact_model(self, state_values: Tuple, action_values: Tuple):
        state_values, reward = self.model(state_values, action_values)
        state_cfvalues = tuple(var.cfvalue for var in self.out_state_vars)
        cfreward = tuple(var.cfvalue for var in self.reward_vars)
        return state_values, reward, state_cfvalues, cfreward
