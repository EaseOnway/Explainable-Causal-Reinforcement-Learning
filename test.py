from learning.explain import Explainner
from learning.env_model import CausalEnvModel
from learning.planning import Actor
from learning.buffer import Buffer
from baselines import BaselineExplainner
from alg import Experiment
from core import Tag
import json


class Test(Experiment):
    use_existing_path = True

    def make_title(self):
        return 'test'

    def setup(self):
        super().setup()
        self.model = CausalEnvModel(self.context)
        self.actor = Actor(self.context)
        self.actor.load(self.path / 'actor.nn')
        self.model.load(self.path / 'explain-env-model.nn')
        self.explainer = Explainner(self.actor, self.model)
        # self.baseline = BaselineExplainner(self.actor)
        # self.baseline.load(self.path / 'explain-baseline.nn')

        with open(self.path / 'explain-causal-graph.json', 'r') as f:
            graph = json.load(f)
            self.model.load_graph(graph)

        self.trajectory = Buffer(self.context, 
             max_size=(self.config.rl.max_episode_length or 1000))

    def main(self):
        episode = 0
        i = 0
        while True:
            length = input()
            explain = False

            if length == 'q':
                break
            elif length == 'r':
                episode += 1
                i = 0
                self.env.reset()
            elif length == 'e':
                explain = True

            try:
                length = int(length)
            except ValueError:
                length = 1

            for _ in range(length):
                a = self.actor.act(self.env.current_state)
                transition = self.env.step(a)
                self.trajectory.write(transition.variables, transition.reward,
                                      Tag.encode(transition.terminated, False, False))
                variables = transition.variables

                print(f"episode {episode}, step {i}:")
                print(f"| state:")
                for name in self.env.names_s:
                    print(f"| | {name} = {variables[name]}")
                print(f"| action:")
                for name in self.env.names_a:
                    print(f"| | {name} = {variables[name]}")
                print(f"| next state:")
                for name in self.env.names_next_s:
                    print(f"| | {name} = {variables[name]}")
                print(f"| outcome:")
                for name in self.env.names_o:
                    print(f"| | {name} = {variables[name]}")
                print(f"| reward = {transition.reward}")
                if len(transition.info) > 0:
                    print(f"| info:")
                    for k, v in transition.info.items():
                        (f"| | {k} = {v}")
                
                if explain:
                    self.explainer.why(
                        self.trajectory.transitions[-5:],
                        maxlen=5, thres=0.1, mode=True,
                        plotfile=str(self._file_path('causal-chain')),
                        # to={'getting close'}
                    )
                    # self.baseline.why(
                    #     self.env.state_of(transition.variables),
                    #     self.env.action_of(transition.variables)
                    # )
                if transition.terminated:
                    print(f"episode {episode} terminated.")
                    episode += 1
                    i = 0
                else:
                    i += 1

        self.env.reset()

Experiment.register('test', Test)
# Experiment.run(['test', r'--path=experiments\buildmarine\test'])
# Experiment.run(['test', r'--path=experiments\lunarlander\model-based\test'])
Experiment.run(['test', r'--path=E:\OneDrive\工作\temp'])
