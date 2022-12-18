from learning.explain import Explainner
from learning.env_model import AttnCausalModel
from learning.planning import Actor
from learning.buffer import Buffer
from baselines import BaselineExplainner
from alg import Experiment
from core import Tag
import json


class Test(Experiment):

    def make_title(self):
        return 'test'
    
    @classmethod
    def init_parser(cls, parser):
        super().init_parser(parser)

        parser.add_argument('--actor', type=str, required=True)
        parser.add_argument('--env-model', type=str, required=True)
        parser.add_argument('--graph', type=str, required=True)
        parser.add_argument('--baseline', type=str, required=True)
    def setup(self):
        super().setup()
        self.model = AttnCausalModel(self.context)
        self.actor = Actor(self.context)
        self.actor.load(self.args.actor)
        self.model.load(self.args.env_model)
        self.explainer = Explainner(self.actor, self.model)
        self.baseline = BaselineExplainner(self.actor)
        self.baseline.load(self.args.baseline)

        with open(self.args.graph, 'r') as f:
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
                        maxlen=5, thres=0.2, mode=True)
                    self.baseline.why(
                        self.env.state_of(transition.variables),
                        self.env.action_of(transition.variables)
                    )
                if transition.terminated:
                    print(f"episode {episode} terminated.")
                    episode += 1
                    i = 0
                else:
                    i += 1

        self.env.reset()


test = Test(['cancer',
             r"--actor=experiments\cancer\model-free\run-7\actor.nn",
             r"--env-model=experiments\cancer\explain\run-2\env-model-0.nn",
             r"--graph=experiments\cancer\explain\run-2\causal-graph.json",
             r"--baseline=experiments\cancer\explain\run-2\baselines.nn"])
test.execute()
