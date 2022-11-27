import os
from typing import Any, Callable, Dict, List, Literal, Optional, final, Tuple
import numpy as np
import random
import torch
from scipy.stats import chi2


import tensorboardX
from tensorboard.backend.event_processing import event_accumulator

from core import Batch, Transitions, Tag
from .buffer import Buffer
from .causal_discovery import discover, update
from .causal_model import CausalNet, CausalModel
from .config import Config
from .base import Configured
from .planning import PPO
from utils import Log, RewardScaling
from utils.typings import ParentDict, NamedArrays
import utils


_LL = 'loglikelihood'
_NLL_LOSS = 'NLL loss'
_ACTOR_LOSS = 'actor loss'
_CRITIC_LOSS = 'critic loss'
_ACTOR_ENTROPY = 'actor entropy'
_REWARD = 'reward'
_EXP_ROOT = './experiments/'
_CAUSAL_GRAPH = 'causal.graph'
_PPO_ACTOR = 'ppo.actor'
_PPO_CRITIC = 'ppo.critic'
_CAUSAL_NET = 'causal.net'
_REWARD_SCALING = 'reward_scaling'
_SAVED_STATE_DICT = 'saved_state_dict'
_SAVED_CONFIG = 'config.txt'
_RETURN = 'return'


class Train(Configured):

    @staticmethod
    def set_seed(seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def __init__(self, config: Config, name: str,
                 showinfo: Literal[None, 'brief', 'verbose', 'plot'] = 'verbose'):
        super().__init__(config)
        print("Using following configuration:")
        print(self.config)

        # arguments
        self.causal_args = self.config.causal_args
        self.rl_args = self.config.rl_args
        self.name = name
        self.dir = _EXP_ROOT + '/' + str(self.env) + '/' + self.name + '/'
        self.show_loss = (showinfo is not None)
        self.show_detail = (showinfo == 'verbose' or showinfo == 'plot')
        self.show_plot = (showinfo == 'plot')

        # causal graph, causal model, and model optimizer
        self.__causal_graph: ParentDict
        self.causnet = CausalNet(config)
        self.opt = self.F.get_optmizer(self.causal_args.optim_args, self.causnet)

        # planning algorithm
        self.ppo = PPO(config)

        # buffer for causal model
        self.buffer_m = Buffer(config, self.causal_args.buffer_size)

        # buffer for planning
        self.buffer_p = Buffer(config, config.rl_args.buffer_size)

        # statistics for reward scaling
        self.__reward_scaling = RewardScaling(self.rl_args.discount)
        
        # declear runtime variables
        self.__n_sample: int
        self.__episodic_return: float
        self.__run_dir: str
        self.__writer: tensorboardX.SummaryWriter
    
    @property
    @final
    def causal_graph(self):
        return self.__causal_graph

    @causal_graph.setter
    @final
    def causal_graph(self, graph: ParentDict):
        self.__causal_graph = graph
        self.causnet.load_graph(self.__causal_graph)
    
    def state_dict(self):
        return {_PPO_ACTOR: self.ppo.actor.state_dict(),
                _PPO_CRITIC: self.ppo.critic.state_dict(),
                _CAUSAL_NET: self.causnet.state_dict(),
                _CAUSAL_GRAPH: self.__causal_graph,
                _REWARD_SCALING: self.__reward_scaling.state_dict()}
    
    def load_state_dict(self, dic: Dict[str, Any]):
        self.ppo.actor.load_state_dict(dic[_PPO_ACTOR])
        self.ppo.critic.load_state_dict(dic[_PPO_CRITIC])
        self.causnet.load_state_dict(dic[_CAUSAL_NET])
        self.__reward_scaling.load_state_dict(dic[_REWARD_SCALING])
        self.causal_graph = dic[_CAUSAL_GRAPH]

    def __collect(self, buffer: Buffer, n_sample: int, random = False):
        '''collect real-world samples into the buffer, and compute returns'''

        log = Log()
        self.env.reset()
        initiated = True
        i_step: int = 0
        max_len = self.causal_args.maxlen_truth

        for i_sample in range(n_sample):
            # interact with the environment
            if random:
                a = self.env.random_action()
            else:
                a = self.ppo.act(self.env.current_state, False)

            transition = self.env.step(a)

            reward = transition.reward
            terminated = transition.terminated
            truncated = (i_step == max_len)

            # record information
            self.__episodic_return += reward
            log[_REWARD] = reward
            if truncated or terminated:
                initiated = True
                i_step = 0
                log[_RETURN] = self.__episodic_return
                self.__episodic_return = 0.
            else:
                initiated = False
                i_step += 1

            # reset environment
            if truncated:
                self.env.reset()
            
            # truncate the last sample
            if i_sample == n_sample - 1:
                truncated = True

            # write buffer
            tagcode = Tag.encode(terminated, truncated, initiated)
            if self.rl_args.use_reward_scaling:
                reward = self.__reward_scaling(reward, (truncated or terminated))
            
            buffer.write(transition.variables, reward, tagcode)
        
        self.__n_sample += n_sample
        return log

    def __dream(self, buffer: Buffer, n_sample: int):
        '''generate samples into the buffer using the environment model'''
        buffer = self.buffer_p
        log = Log()

        max_len = self.causal_args.maxlen_dream
        batchsize = self.causal_args.dream_batch_size
        env_m = CausalModel(self.causnet, self.buffer_m, max_len)
        env_m.reset(batchsize)
        
        tr: List[Transitions] = []
        
        for i_sample in range(0, n_sample, batchsize):
            with torch.no_grad():
                s = env_m.current_state
                a = self.ppo.actor.forward(s).sample().kapply(self.label2raw)
                tran = env_m.step(a)
            tr.append(tran)
            i_sample += tran.n
        
        i_sample = 0
        for i in range(batchsize):
            rj = range(0, min(len(tr), n_sample-i_sample))
            i_sample += rj.stop

            data = {name: torch.stack([tr[j][name][i] for j in rj])
                    for name in self.env.names_all}
            reward = torch.stack([tr[j].rewards[i]  for j in rj])
            code = torch.stack([tr[j].tagcode[i] for j in rj])
            code[-1] |= Tag.TRUNCATED.mask

            for r in reward:
                log[_REWARD] = float(r)

            if self.rl_args.use_reward_scaling:
                reward /= self.__reward_scaling.std

            t = Transitions(data, reward, code)
            buffer.append(t)
            
        assert i_sample == n_sample
        return log

    def __init_params(self):
        self.ppo.actor.init_parameters()
        self.ppo.critic.init_parameters()
        self.causnet.init_parameters()
        self.causal_graph = {j: {i for i in self.env.names_inputs
                                 if utils.Random.event(self.causal_args.prior)}
                             for j in self.env.names_outputs}

    def plot_causal_graph(self, format='png'):
        return utils.visualize.plot_digraph(
            self.env.names_inputs + self.env.names_outputs,
            self.__causal_graph, format=format)  # type: ignore

    def __fit_batch(self, transitions: Transitions, eval=False):
        self.causnet.train(not eval)
        lls = self.causnet.get_loglikeli_dic(transitions)
        ll = self.causnet.loglikelihood(lls)
        loss = -ll
        if not eval:
            loss.backward()
            self.F.optim_step(self.causal_args.optim_args,
                              self.causnet, self.opt)
        return float(loss), {k: float(e) for k, e in lls.items()}

    def __fit_epoch(self, log: Log, eval=False):
        '''
        train network with fixed causal graph.
        '''
        args = self.causal_args.optim_args
        for batch in self.buffer_m.epoch(args.batchsize):
            loss, lls = self.__fit_batch(batch, eval)
            log[_NLL_LOSS] = loss
            for k, ll in lls.items():
                log[_LL, k] = ll

    def __show_fit_log(self, log: Log):
        Log.figure(figsize=(12, 5))  # type: ignore
        Log.subplot(121, title=_NLL_LOSS)
        log[_NLL_LOSS].plot(color='k')
        Log.subplot(122, title=_LL)
        log[_LL].plots(self.env.names_outputs)
        Log.show()
        
    def load(self, path: Optional[str] = None):
        path = path or self.__run_dir + _SAVED_STATE_DICT
        self.load_state_dict(torch.load(path))

    def save(self, path: Optional[str] = None):
        path = path or self.__run_dir + _SAVED_STATE_DICT
        torch.save(self.state_dict(), path)

    def __eval(self):
        log = Log()
        self.__fit_epoch(log, eval=False)
        return log
    
    def __get_data_for_causal_discovery(self) -> NamedArrays:
       temp = self.buffer_m.tensors[:]
       temp = {k: self.raw2input(k, v).numpy() for k, v in temp.items()}
       return temp
    
    def __get_run_dir(self, path: Optional[str] = None):
        if path is None:
            if not os.path.exists(self.dir):
                os.makedirs(self.dir)
            run_names = os.listdir(self.dir)
            i = 0
            while True:
                i += 1
                run_name = "run-%d" % i
                if run_name not in run_names:
                    break
            path = self.dir + run_name + '/'
            os.makedirs(path)
        else:
            if len(path) == 0:
                path = './'
            elif path[-1] != '/' or path[-1] != '\\':
                path += '/'
            if not os.path.exists(path):
                os.makedirs(path)
        return path
    
    def __resume_nsample(self, path: str):
        e = event_accumulator.EventAccumulator(path)
        e.Reload()
        return int(e.scalars.Items('return')[-1].step)
    
    def init_run(self, dir: Optional[str] = None, resume=False,
                 n_sample: Optional[int] = None):
        path = self.__run_dir = self.__get_run_dir(dir)
        self.config.to_txt(path + _SAVED_CONFIG)

        if resume:
            self.load()
            if n_sample is None:
                self.__n_sample = self.__resume_nsample(path)
                self.__writer = tensorboardX.SummaryWriter(path)
            else:
                self.__n_sample = n_sample
                self.__writer = tensorboardX.SummaryWriter(path, purge_step=n_sample)
        else:
            self.__n_sample = 0
            self.__init_params()
            self.save()
            self.__writer = tensorboardX.SummaryWriter(path)

        self.__episodic_return = 0.
        
        self.env.reset()
        self.buffer_m.clear()
        self.buffer_p.clear()

    def warmup(self, n_sample: int, random=False):
        return self.__collect(self.buffer_m, n_sample, random=random)
    
    def fit(self, n_epoch: int):
        # fit causal equation
        train_log = Log()
        for i_epoch in range(n_epoch):
            self.__fit_epoch(train_log, eval=False)
            if self.show_detail:
                print(f"fit epoch {i_epoch} done.")
        
        # evaluate
        eval_log = self.__eval()
        
        # show info
        if self.show_loss:
            print(f"fitting loss:\t{eval_log[_NLL_LOSS].mean}")
        if self.show_detail:
            for k in self.env.names_outputs:
                print(f"log-likelihood of '{k}':\t{eval_log[_LL, k].mean}")
        if self.show_plot:
            self.__show_fit_log(train_log)
        
        return train_log, eval_log

    def causal_reasoning(self, n_epoch: int):
        # causal discovery
        data = self.__get_data_for_causal_discovery()
        self.causal_graph = discover(data, self.env,
                                     self.causal_args.pthres_independent,
                                     self.show_detail,
                                     self.causal_args.n_jobs_fcit)
        train_log, eval_log = self.fit(n_epoch)
        
        # save
        self.save()
        print('Done.')

        return train_log, eval_log

    def __step_policy_model_based(self, i_step: int):
        # collect true samples
        log_step = self.__collect(self.buffer_m, self.causal_args.n_true_sample)
        true_reward = log_step[_REWARD].mean
        true_return = log_step[_RETURN].mean

        # fit causal equation
        if not (self.ablations.graph_fixed or self.ablations.graph_offline) \
                and i_step % self.causal_args.interval_graph_update == 0:
            _, fit_eval = self.causal_reasoning(
                self.causal_args.n_epoch_fit_new_graph)
        else:
            # fit causal equation
            _, fit_eval = self.fit(self.causal_args.n_epoch_fit)

        # planning
        plan_log = Log()
        for i_round in range(self.rl_args.n_round_model_based):
            # dream samples
            self.buffer_p.clear()
            dream_log = self.__dream(self.buffer_p, self.buffer_p.max_size)
            plan_log[_REWARD] = dream_log[_REWARD].mean

            # planning
            loss = self.ppo.optimize(self.buffer_p)
            plan_log[_ACTOR_LOSS] = actor_loss = loss['actor'].mean
            plan_log[_CRITIC_LOSS] = critic_loss = loss['critic'].mean
            plan_log[_ACTOR_ENTROPY] = actor_entropy = loss['entropy'].mean
            if self.show_loss:
                print("round %d: actor loss = %f, critic loss= %f, actor entropy = %f" %
                        (i_round, actor_loss, critic_loss, actor_entropy))

        dream_reward = plan_log[_REWARD].data[0]
        actor_loss = plan_log[_ACTOR_LOSS].data[-1]
        critic_loss = plan_log[_CRITIC_LOSS].data[-1]
        actor_entropy = plan_log[_ACTOR_ENTROPY].data[-1]
        
        # write summary
        writer = self.__writer
        writer.add_scalar('reward', true_reward, self.__n_sample)
        if not np.isnan(true_return):
            writer.add_scalar('return', true_return, self.__n_sample)
        writer.add_scalar('reward_dreamed', dream_reward, self.__n_sample)
        writer.add_scalar('actor_loss', actor_loss, self.__n_sample)
        writer.add_scalar('critic_loss', critic_loss, self.__n_sample)
        writer.add_scalar('actor_entropy', actor_entropy, self.__n_sample)
        writer.add_scalar('fitting_loss', fit_eval[_NLL_LOSS].mean, self.__n_sample)
        writer.add_scalars('log_likelihood',
                           {k: fit_eval[_LL, k].mean
                            for k in self.env.names_outputs},
                           self.__n_sample)

        # show info
        if self.show_loss:
            print(f"actor loss:\t{actor_loss}")
            print(f"critic loss:\t{critic_loss}")
        if self.show_detail:
            print(f"episodic return:\t{true_return}")
            print(f"mean reward:\t{true_reward} (truth); {dream_reward} (dream)")
            print(f"actor entropy:\t{actor_entropy}")
    
    def __step_policy_model_free(self, i_step: int):
        # collect samples
        self.buffer_p.clear()
        log = self.__collect(self.buffer_p, self.buffer_p.max_size)
        true_reward = log[_REWARD].mean
        true_return = log[_RETURN].mean
        self.buffer_m.append(
            self.buffer_p.sample_batch(self.causal_args.n_true_sample))

        # planning
        plan_loss = self.ppo.optimize(self.buffer_p)
        actor_loss = plan_loss['actor'].mean
        critic_loss = plan_loss['critic'].mean
        actor_entropy = plan_loss['entropy'].mean

        # write summary
        writer = self.__writer
        writer.add_scalar('reward', true_reward, self.__n_sample)
        if not np.isnan(true_return):
            writer.add_scalar('return', true_return, self.__n_sample)
        writer.add_scalar('actor_loss', actor_loss, self.__n_sample)
        writer.add_scalar('critic_loss', critic_loss, self.__n_sample)
        writer.add_scalar('actor_entropy', actor_entropy, self.__n_sample)

        # show info
        if self.show_loss:
            print(f"actor loss:\t{plan_loss['actor'].mean}")
            print(f"critic loss:\t{plan_loss['critic'].mean}")
        if self.show_detail:
            print(f"episodic return:\t{true_return}")
            print(f"mean reward:\t{true_reward}")
            print(f"actor entropy:\t{actor_entropy}")
        if self.show_plot:
            self.ppo.show_loss(plan_loss)

    def iter_policy(self, n_step: int, model_based=False):
        for i in range(0, n_step):
            print(f"step {i}: ")
            if model_based:
                self.__step_policy_model_based(i)
            else:
                self.__step_policy_model_free(i)
            
            self.save()
            print("Done.")
