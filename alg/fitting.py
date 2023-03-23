from typing import Any, Callable, Dict, List, Literal, Optional, final, Tuple
import torch
import numpy as np

from .train import Train

from learning.buffer import Buffer
from learning.planning import Actor
from learning.explain.action_effect import ActionEffect
from learning.causal_discovery import discover_aim

from utils import Log
from utils.typings import ParentDict


_LL = 'loglikelihood'
_NLL_LOSS = 'NLL loss'


class Fitting(Train):
    use_existing_path = False

    @classmethod
    def init_parser(cls, parser):
        super().init_parser(parser)
        parser.add_argument('--actor', type=str, default=None,
            help="path of the saved actor network. by default, the actor will be randomly initialized.")
        parser.add_argument('--test-size', type=int, default=1000)
        parser.add_argument('--train-size', type=int, default=10000)
        parser.add_argument('--n-step', type=int, default=20)
        parser.add_argument('--n-batch', type=int, default=20000)
        parser.add_argument('--ablation', type=str, default=None)
        parser.add_argument('--explore', type=float, default=None,
            help="the probability to take random actions rather than following the actor")
        parser.add_argument('--aim-thres', type=float, default=0.1,
            help="the attention threshold for AIM. Only used when the environment supports groud-truth AIM graph.")


    def make_title(self):
        title = "fitting"
        ablation = self.args.ablation
        if self.args.ablation is None:
            return title
        else:
            return title + '-' + ablation
    
    def configure(self):
        config = super().configure()

        # setting ablation
        ablation = self.args.ablation
        if ablation == 'recur':
            config.ablations.recur = True
        elif ablation == 'offline':
            config.ablations.offline = True
        elif ablation == 'dense':
            config.ablations.dense = True
        elif ablation == 'mlp':
            config.ablations.mlp = True
        elif ablation is not None:
            raise ValueError(f"unsupported ablation: {ablation}")
        
        return config

    def __graph_accuracy(self, graph: ParentDict, true_graph: ParentDict, aim=False):
        g_ = true_graph
        g = graph
        n_correct = 0

        if aim:
            for j in self.env.names_output:
                for i in self.env.names_s:
                    n_correct += int((i in g_[j]) == (i in g[j]))
            n_total = self.env.num_output * self.env.num_s
        else:
            for j in self.env.names_output:
                for i in self.env.names_input:
                    n_correct += int((i in g_[j]) == (i in g[j]))
            n_total = self.env.num_output * self.env.num_input
        return (n_correct / n_total), n_correct, n_total

    def __aim_test(self, attn_thres: float, direct=False):
        env = self.env
        model = self.env_models[0]
        writer = self.writer

        from envs.aimtest import AimTestEnv
        from learning.env_model import CausalEnvModel
        if not isinstance(env, AimTestEnv):
            return
        if not isinstance(model, CausalEnvModel):
            return

        def aim_correct_ness(label: str, gs: List[ParentDict], true_gs: List[ParentDict]):
            n_correct = 0
            n_total = 0
            accs = []
            for a in range(env.N_ACTION):
                g, g_ = gs[a], true_gs[a]
                acc = self.__graph_accuracy(g, g_, aim=True)
                accs.append(acc)
                n_correct += acc[1]
                n_total += acc[2]
            acc = (n_correct/n_total, n_correct, n_total)
            print("- %s AIM: %.5f (%d / %d edges)" % (label, *acc))
            writer.add_scalar('aim-graph-accuracy', acc[0], len(self.buffer_m))
            for a in range(env.N_ACTION):
                acc = accs[a]
                print("- - action = %d: %.5f (%d / %d edges)" % (a, *acc))
        
        true_gs =  [env.action_influence_graph(a) for a in range(env.N_ACTION)]

        if direct:
            print("Performing direct causal discovery for AIM.")
            gs = discover_aim(self._get_data_for_causal_discovery(), self.env,
                              self.config.model.pthres, True, self.config.model.n_jobs_fcit)
            aim_correct_ness("direct", gs, true_gs)
        else:
            g = self.causal_graph
            acc = self.__graph_accuracy(g, env.structural_causal_graph(), aim=False)
            print("- SCM: %.5f (%d / %d edges)" % acc)
            writer.add_scalar('scm-graph-accuracy', acc[0], len(self.buffer_m))

            gs = [ActionEffect(model, {'a': a}).graph(attn_thres)
                  for a in range(env.N_ACTION)]
            
            aim_correct_ness("attention-based", gs, true_gs)

    def setup(self):
        super().setup()
        self.env_models = self.creat_env_models(1)
        self.env_models.init_parameters()
        self.env_model_optimizers = self.env_models.optimizers()

        self.actor = Actor(self.context)
        self.actor.init_parameters()
        if self.args.actor is not None:
            self.actor.load(self.args.actor)
            print(f"successfully loaded actor from {self.args.actor}")
        
        self.buffer_m = Buffer(self.context, self.config.model.buffer_size)
        self.aim_thres: float = self.args.aim_thres

    @property
    def env_model(self):
        return self.env_models[0]
    
    def save_all(self):
        for i, model in enumerate(self.env_models):
            self.save(model.state_dict(), f'env-model-{i}', 'nn')
        if not self.config.ablations.mlp:
            self.save(self.causal_graph, 'causal-graph', 'json')

    def main(self):
        writer = self.writer
        test_size: int = self.args.test_size
        train_size: int = self.args.train_size
        explore_rate = self.args.explore
        n_step: int = self.args.n_step
        n_batch: int = self.args.n_batch

        # collect train samples
        test = Buffer(self.context)

        print("collecting test samples")
        self.collect(test, test_size, explore_rate, False, self.actor)
        self.buffer_m.clear()
        
        interval = max(train_size // n_step, 1)
        for i in range(0, train_size, interval):
            print(f"test ({i + interval}/{train_size}):")
            print("  collecting samples")
            self.collect(self.buffer_m, interval, explore_rate, False, self.actor)

            self.__aim_test(self.aim_thres, direct=True)

            # causal_reasoning
            self.causal_discovery()
            self.env_models.init_parameters()
            self.fit(n_batch, -1)

            self.__aim_test(self.aim_thres, direct=False)

            # eval
            log = Log()
            self.fit_epoch(test, log, eval=True)

            # write summary
            writer.add_scalar('log-likelihood', -log[_NLL_LOSS].mean, len(self.buffer_m))
            writer.add_scalars('log_likelihood_variable',
                {k: log[_LL, k].mean for k in self.env.names_output}, len(self.buffer_m))

            # show info
            print(f"- total log-likelihood:\t{-log[_NLL_LOSS].mean}")
            for k in self.env.names_output:
                print(f"- log-likelihood of '{k}':\t{log[_LL, k].mean}")
        
            # save
            self.save_all()
