import random
import time

import numpy as np

from .helper import *
from ..do_not_touch.contracts import MDPEnv
from ..do_not_touch.result_structures import ValueFunction, PolicyAndValueFunction

np.seterr('raise')


def policy_evaluation(env: MDPEnv, gamma: float, theta: float, pi: dict = None, V: dict = None, env_name: str = '') -> ValueFunction:
    S = env.states()
    A = env.actions()
    R = env.rewards()

    if pi is None:
        pi = {}
        for s in S:
            pi[s] = {a: 1/len(A) for a in A}

    if V is None:
        V = {s: 0 for s in S}

    while True:
        delta = 0
        for s in S:
            v = V[s]
            V[s] = 0
            for a in A:
                for s_p in S:
                    for r_idx, r in enumerate(R):
                        V[s] += pi[s][a] * env.transition_probability(s, a, s_p, r_idx) * (r + gamma * V[s_p])
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break

    return V


def policy_iteration(env: MDPEnv, gamma: float, theta: float, env_name='') -> PolicyAndValueFunction:
    S = env.states()
    A = env.actions()
    R = env.rewards()

    pi = {}
    for s in S:
        pi[s] = {a: 1 / len(A) for a in A}

    V = {s: 0.0 for s in S}

    while True:
        V = policy_evaluation(env, gamma, theta, pi, V)

        policy_stable = True
        for s in S:
            old_state_policy = dict(pi[s])

            best_a = -1
            best_a_score = None

            for a in A:
                a_score = 0.0
                for s_p in S:
                    for r_idx, r in enumerate(R):
                        a_score += env.transition_probability(s, a, s_p, r_idx) * (r + gamma * V[s_p])

                if best_a_score is None or best_a_score < a_score:
                    best_a = a
                    best_a_score = a_score

                pi[s][a] = 0.0
            pi[s][best_a] = 1.0

            if old_state_policy != pi[s]:
                policy_stable = False
        if policy_stable:
            break

    pi_v = PolicyAndValueFunction(pi, V)
    weight = f'{WEIGHT_PATH}{env_name}_policy_iteration_{pi_v.__class__.__name__}'

    save_to_pickle(pi_v, weight)

    return pi_v


def value_iteration(env: MDPEnv, gamma: float, theta: float, env_name='') -> PolicyAndValueFunction:
    S = env.states()
    A = env.actions()
    R = env.rewards()

    pi = {}
    for s in S:
        pi[s] = {}
        for a in A:
            pi[s][a] = 0
        pi[s][random.randint(0, len(A) - 1)] = 1.0

    V = {s: 0 for s in S}

    while True:
        delta = 0
        for s in S:
            v = V[s]
            V[s] = 0
            Vs_a = []
            for a in A:
                vs = 0
                for s_p in S:
                    for r_idx, r in enumerate(R):
                        vs += env.transition_probability(s, a, s_p, r_idx) * (r + gamma * V[s_p])
                Vs_a.append(vs)
            V[s] = max(Vs_a)

            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break

    for s in S:
        vs_a = []
        for a in A:
            vs = 0
            for s_p in S:
                for r_idx, r in enumerate(R):
                    vs += env.transition_probability(s, a, s_p, r_idx) * (r + gamma * V[s_p])
            vs_a.append(vs)
            pi[s][a] = 0.0
        pi[s][np.argmax(vs_a)] = 1.0

    pi_v = PolicyAndValueFunction(pi, V)
    weight = f'{WEIGHT_PATH}{env_name}_value_iteration_{pi_v.__class__.__name__}'

    save_to_pickle(pi_v, weight)

    return pi_v
