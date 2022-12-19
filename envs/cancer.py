import numpy as np
from core import Env
from core import ContinuousNormal, Binary


P = 'P'  # proliferate tissue
Q = 'Q'  # quiescent tissue
C = 'C'  # concentration
QP = 'Q_p'  # damaged tissue
NEXT = {name: Env.name_next(name) for name in (Q, P, C, QP)}

TREAT = 'treat'
COST = 'cost'

class Cancer(Env):

    @classmethod
    def init_parser(cls, parser):
        parser.add_argument("--price", type=float, default=1.)
        parser.add_argument("--env-noise", type=float, default=0.)
    
    def launch(self):
        pass

    def define(self, args):
        _def = Env.Definition()
    
        ''' parameters '''
        self.kde = 0.24
        self.lambda_p = 0.121
        self.k_qpp = 0.0031
        self.k_pq = 0.0295
        self.gamma = 0.729
        self.delta_qp = 0.00867
        self.k = 100
        self.price: float = args.price
        self.env_noise: float = args.env_noise

        _def.state(Q, ContinuousNormal(scale=None))
        _def.state(P, ContinuousNormal(scale=None))
        _def.state(C, ContinuousNormal(scale=None))
        _def.state(QP, ContinuousNormal(scale=None))
        _def.action(TREAT, Binary())
        _def.outcome(COST, ContinuousNormal(scale=None))

        _def.reward('tumor reduction', [Q, P, QP, NEXT[Q], NEXT[P], NEXT[QP]],
            lambda q, p, qp, q_, p_, qp_: (q + p + qp) - (q_ + p_ + qp_))
        _def.reward('cost', [COST], lambda c: -c)

        return _def

    def init_episode(self):
        return {C: 0,
                P: 7.13,
                Q: 41.2,
                QP: 0}

    def __noised(self, x):
        return x * (1 + self.env_noise * np.random.randn())

    def transit(self, actions):
        c = self.current_state[C]
        p = self.current_state[P]
        q = self.current_state[Q]
        qp = self.current_state[QP]
        treat = actions[TREAT]
        
        p_star = p + q + qp
        delta_c = - self.kde * c
        delta_p = self.lambda_p * p * (1-p_star/self.k) + self.k_qpp * qp \
            - self.k_pq * p - self.gamma * c * self.kde * p
        delta_q = self.k_pq * p - self.gamma * c * self.kde * q
        delta_qp = self.gamma * c * self.kde * q - self.k_qpp * qp \
            - self.delta_qp * qp
        
        c_ = c + delta_c
        if treat:
            cost = 3. - c_
            c_ = 3.
        else:
            cost = 0.
        
        out = {}
        out[NEXT[P]] = self.__noised(p + delta_p) 
        out[NEXT[Q]] = self.__noised(q + delta_q)
        out[NEXT[QP]] = self.__noised(qp + delta_qp)
        out[NEXT[C]] = self.__noised(c_)
        out[COST] = self.__noised(cost * self.price)
        return out, {}

    def terminated(self, variables) -> bool:
        return False
    
    def random_action(self):
        return {TREAT: np.random.rand() < 0.5}

    @property
    def true_causal_graph(self):
        return {
            NEXT[P]: (P, Q, QP, C),
            NEXT[C]: (TREAT, C),
            NEXT[Q]: (P, C, Q),
            NEXT[QP]: (QP, Q, C),
            COST: (C, TREAT)}
