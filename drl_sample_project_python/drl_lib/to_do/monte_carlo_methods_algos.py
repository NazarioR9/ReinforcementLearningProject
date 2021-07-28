import numpy as np

from .helper import *
from ..do_not_touch.contracts import SingleAgentEnv
from ..do_not_touch.result_structures import PolicyAndActionValueFunction


def monte_carlo_es(env: SingleAgentEnv, gamma: float, max_iter: int, env_name: str = '') -> PolicyAndActionValueFunction:
    # *********** Helper *********
    name = f'{env_name}_monte_carlo_es'
    print_every_n_episodes = 10
    reward_tracker = RewardTracker(name, 5 * print_every_n_episodes)
    # *********** Helper *********

    pi = {}
    q = {}
    returns = {}

    for _ in range(max_iter):
        env.reset()

        R = []
        S = []
        A = []

        while not env.is_game_over():
            s = env.state_id()
            S.append(s)

            available_actions = env.available_actions_ids()

            if s not in pi:
                pi[s] = {}
                q[s] = {}
                returns[s] = {}
                for a in available_actions:
                    pi[s][a] = 1. / len(available_actions)
                    q[s][a] = 0.  # Not sure if optimal
                    returns[s][a] = []

            chosen_action = np.random.choice(list(pi[s].keys()), 1, False, list(pi[s].values()))[0]

            A.append(chosen_action)
            old_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - old_score
            R.append(r)

        # *********** Helper *********
        reward_tracker(env.score())
        # *********** Helper *********

        G = 0
        for t in reversed(range(len(S))):
            G = gamma * G + R[t]

            s_t = S[t]
            a_t = A[t]
            found = False

            for prev_s, prev_a in zip(S[:t], A[:t]):
                if prev_s == s_t and prev_a == a_t:
                    found = True
                    break

            if found:
                continue

            returns[s_t][a_t].append(G)  # TODO use mean/n tricks
            q[s_t][a_t] = np.mean(returns[s_t][a_t])

            a_t_opt = list(q[s_t])[np.argmax(list(q[s_t].values()))]

            for a_key, q_s_a in q[s_t].items():
                if a_key == a_t_opt:
                    pi[s_t][a_key] = 1.0
                else:
                    pi[s_t][a_key] = 0.0

    pi_q = PolicyAndActionValueFunction(pi, q)
    weight = f'{WEIGHT_PATH}{env_name}_monte_carlo_es_{pi_q.__class__.__name__}'
    save_to_pickle(pi_q, weight)

    reward_tracker.save_to_image()
    reward_tracker.save_to_file()

    return pi_q


def on_policy_visit_monte_carlo_control(env: SingleAgentEnv, eps: float, gamma: float,
                                        max_iter: int, env_name: str = '') -> PolicyAndActionValueFunction:
    # *********** Helper *********
    name = f'{env_name}_monte_carlo_on_pvc'
    print_every_n_episodes = 10
    reward_tracker = RewardTracker(name, 5*print_every_n_episodes)
    # *********** Helper *********

    pi = {}
    q = {}
    returns = {}

    for it in range(max_iter):
        env.reset()

        R = []
        S = []
        A = []

        while not env.is_game_over():
            s = env.state_id()
            S.append(s)

            available_actions = env.available_actions_ids()

            if s not in pi:
                pi[s] = {}
                q[s] = {}
                returns[s] = {}
                for a in available_actions:
                    pi[s][a] = 1. / len(available_actions)
                    q[s][a] = 0.  # Not sure if optimal
                    returns[s][a] = []

            chosen_action = np.random.choice(list(pi[s].keys()), 1, False, list(pi[s].values()))[0]

            A.append(chosen_action)
            old_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - old_score
            R.append(r)

        # *********** Helper *********
        reward_tracker(env.score())
        # *********** Helper *********

        G = 0
        for t in reversed(range(len(S))):
            G = gamma * G + R[t]

            s_t = S[t]
            a_t = A[t]
            found = False

            for prev_s, prev_a in zip(S[:t], A[:t]):
                if prev_s == s_t and prev_a == a_t:
                    found = True
                    break

            if found:
                continue

            returns[s_t][a_t].append(G)  # TODO use mean/n tricks
            q[s_t][a_t] = np.mean(returns[s_t][a_t])

            a_t_opt = list(q[s_t])[np.argmax(list(q[s_t].values()))]
            available_actions_t_count = len(q[s_t])

            for a_key, q_s_a in q[s_t].items():
                if a_key == a_t_opt:
                    pi[s_t][a_key] = 1 - eps + (eps / available_actions_t_count)
                else:
                    pi[s_t][a_key] = eps / available_actions_t_count

    pi_q = PolicyAndActionValueFunction(pi, q)
    weight = f'{WEIGHT_PATH}{env_name}_monte_carlo_on_pvc_{pi_q.__class__.__name__}'
    save_to_pickle(pi_q, weight)

    reward_tracker.save_to_image()
    reward_tracker.save_to_file()

    return pi_q


def off_policy_visit_monte_carlo_control(env: SingleAgentEnv, gamma: float,
                                         max_iter: int, env_name: str = '') -> PolicyAndActionValueFunction:
    # *********** Helper *********
    name = f'{env_name}_monte_carlo_off_pvc'
    print_every_n_episodes = 10
    reward_tracker = RewardTracker(name, 5 * print_every_n_episodes)
    # *********** Helper *********

    pi = {}
    q = {}
    C = {}
    b = {}

    for it in range(max_iter):
        env.reset()

        R = []
        S = []
        A = []

        while not env.is_game_over():
            s = env.state_id()
            S.append(s)

            available_actions = env.available_actions_ids()

            if s not in C:
                C[s] = {}
                for a in available_actions:
                    C[s][a] = 0

            if s not in pi:
                pi[s] = {}
                b[s] = {}
                q[s] = {}
                for a in available_actions:
                    pi[s][a] = 1. / len(available_actions)
                    b[s][a] = 1. / len(available_actions)
                    q[s][a] = 0.  # Not sure if optimal

            chosen_action = np.random.choice(list(b[s].keys()), 1, False, list(b[s].values()))[0]

            A.append(chosen_action)
            old_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - old_score
            R.append(r)

        # *********** Helper *********
        reward_tracker(env.score())
        # *********** Helper *********

        G = 0
        W = 1
        for t in reversed(range(len(S))):
            G = gamma * G + R[t]

            s_t = S[t]
            a_t = A[t]

            C[s_t][a_t] += W
            q[s_t][a_t] += (W / C[s_t][a_t]) * (G - q[s_t][a_t])

            a_t_opt = list(q[s_t])[np.argmax(list(q[s_t].values()))]

            for a_key, q_s_a in q[s_t].items():
                if a_key == a_t_opt:
                    pi[s_t][a_key] = 1.0
                else:
                    pi[s_t][a_key] = 0.0

            if a_t != a_t_opt:
                break

            W *= 1 / b[s_t][a_t]

    pi_q = PolicyAndActionValueFunction(pi, q)
    weight = f'{WEIGHT_PATH}{env_name}_monte_carlo_off_pvc_{pi_q.__class__.__name__}'
    save_to_pickle(pi_q, weight)

    reward_tracker.save_to_image()
    reward_tracker.save_to_file()

    return pi_q
