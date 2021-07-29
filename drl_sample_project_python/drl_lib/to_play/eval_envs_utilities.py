import random

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from drl_sample_project_python.drl_lib.do_not_touch.contracts import SingleAgentEnv
from drl_sample_project_python.drl_lib.do_not_touch.mdp_env_wrapper import Env1
from drl_sample_project_python.drl_lib.do_not_touch.result_structures import PolicyAndValueFunction, \
    PolicyAndActionValueFunction
from drl_sample_project_python.drl_lib.do_not_touch.single_agent_env_wrapper import Env3, Env2
from drl_sample_project_python.drl_lib.to_do.deep_reinforcement_learning_algos import DeepQNetwork
from drl_sample_project_python.drl_lib.to_do.envs import LineWorld, GridWorld, TicTacToe, DeepTicTacToe, PacMan
from drl_sample_project_python.drl_lib.to_do.helper import load_from_pickle, WEIGHT_PATH, PLOT_PATH, SLIDE_PATH

"""
Compute The percentage of victories per Env/Method (Algorithm), after training, on n=3000 episodes.
"""


def play_on_lw(pi_v: PolicyAndValueFunction, max_episodes=3000):
    env = LineWorld(7)
    count = 0

    for _ in range(max_episodes):
        s = int(env.cells_count / 2)
        while not env.is_state_terminal(s):
            a = np.argmax(list(pi_v.pi[s].values()))

            if a == 0:
                s -= 1
            else:
                s += 1

        if s == (env.cells_count - 1):
            count += 1

    return count / max_episodes


def play_on_gw(pi_v: PolicyAndValueFunction, max_episodes=3000):
    env = GridWorld(5)
    count = 0

    for _ in range(max_episodes):
        s = 0
        while not env.is_state_terminal(s):
            a = np.argmax(list(pi_v.pi[s].values()))

            if a == 0:
                s -= env.grid_count
            elif a == 1:
                s += env.grid_count
            elif a == 2:
                s -= 1
            else:
                s += 1

        if s == (env.max_cells - 1):
            count += 1

    return count / max_episodes


def play_on_env1(pi_v: PolicyAndValueFunction, max_episodes=3000):
    env = Env1()
    count = 0

    def scorer(action1, action2):
        if (action1 == 0 and action2 == 1) or (action1 == 1 and action2 == 2) or (action1 == 2 and action2 == 0):
            return -1
        if (action1 == 0 and action2 == 2) or (action1 == 1 and action2 == 0) or (action1 == 2 and action2 == 1):
            return 1
        return 0

    for _ in range(max_episodes):
        s = 0

        a = np.argmax(list(pi_v.pi[s].values()))
        r = random.choice(env.actions())

        score = scorer(a, r)

        if score == 0:
            s += 1

            r = a
            a = np.argmax(list(pi_v.pi[s].values()))

            score = scorer(a, r)

        if score == 1:
            count += 1

    return count / max_episodes


def play_with_sa_env(pi_q: PolicyAndActionValueFunction, callable_env, max_episodes=3000):
    count = 0

    for _ in range(max_episodes):
        env = callable_env()
        while not env.is_game_over():
            s = env.state_id()

            if s not in pi_q.pi:
                a = random.choice(env.available_actions_ids())
            else:
                a = list(pi_q.pi[s].keys())[np.argmax(list(pi_q.pi[s].values()))]

            env.act_with_action_id(a)

        if env.score() == 1.0:
            count += 1

    return count / max_episodes


def play_on_deep(q: DeepQNetwork, callable_env, max_episodes=3000):
    count = 0

    for _ in tqdm(range(max_episodes)):

        env = callable_env()
        state_description_length = env.state_description_length()
        max_actions_count = env.max_actions_count()

        while not env.is_game_over():
            s = env.state_description()
            available_actions = env.available_actions_ids()
            all_q_inputs = np.zeros((len(available_actions), state_description_length + max_actions_count))

            for i, a in enumerate(available_actions):
                all_q_inputs[i] = np.hstack([s, tf.keras.utils.to_categorical(a, max_actions_count)])
            all_q_values = np.squeeze(q.predict(all_q_inputs))

            chosen_action = available_actions[np.argmax(all_q_values)]

            env.act_with_action_id(chosen_action)

        if env.score() == 1.0:
            count += 1

    return count / max_episodes


def play_on_deep_pacman(q: DeepQNetwork, callable_env, max_episodes=3000):
    running_score = 0
    max_score = 0

    for _ in tqdm(range(max_episodes)):

        env = callable_env()
        state_description_length = env.state_description_length()
        max_actions_count = env.max_actions_count()

        if max_score == 0:
            max_score = len(np.where(env.map == env.entity_ids['food'])[0]) * max_episodes

        while not env.is_game_over():
            s = env.state_description()
            available_actions = env.available_actions_ids()
            all_q_inputs = np.zeros((len(available_actions), state_description_length + max_actions_count))

            for i, a in enumerate(available_actions):
                all_q_inputs[i] = np.hstack([s, tf.keras.utils.to_categorical(a, max_actions_count)])
            all_q_values = np.squeeze(q.predict(all_q_inputs))

            chosen_action = available_actions[np.argmax(all_q_values)]

            env.act_with_action_id(chosen_action)

        running_score += env.score()

    return running_score / max_score


def eval_on_dp_envs():
    envs = {
        "lineworld": play_on_lw,
        "gridworld": play_on_gw,
        "env1": play_on_env1
    }
    methods = ['policy_iteration', 'value_iteration']

    results = {}

    for env_name, estimator in envs.items():
        results[env_name] = {}
        for method in methods:
            pi_v = load_from_pickle(f'{WEIGHT_PATH}{env_name}_{method}_PolicyAndValueFunction')

            results[env_name][method] = estimator(pi_v)

    print("Evaluation on Dynamic Programming")
    print(results)
    print()


def eval_on_mc_envs():
    envs = {
        "tictactoe": TicTacToe,
        "env2": Env2,
    }
    methods = ['monte_carlo_es', 'monte_carlo_off_pvc', 'monte_carlo_on_pvc']

    results = {}

    for env_name, callable_env in envs.items():
        results[env_name] = {}
        for method in methods:
            pi_q = load_from_pickle(f'{WEIGHT_PATH}{env_name}_{method}_PolicyAndActionValueFunction')
            results[env_name][method] = play_with_sa_env(pi_q, callable_env)

    print("Evaluation on Monte Carlo")
    print(results)
    print()


def eval_on_td_envs():
    envs = {
        "tictactoe": TicTacToe,
        "env3": Env3,
    }
    methods = ['sarsa', 'q_learning', 'expected_sarsa']

    results = {}

    for env_name, callable_env in envs.items():
        results[env_name] = {}
        for method in methods:
            pi_q = load_from_pickle(f'{WEIGHT_PATH}{env_name}_{method}_PolicyAndActionValueFunction')
            results[env_name][method] = play_with_sa_env(pi_q, callable_env)

    print("Evaluation on Temporal Difference")
    print(results)
    print()


def eval_on_drl_envs():
    envs = {
        "tictactoe": (DeepTicTacToe, play_on_deep),
        # "pacman": (PacMan, play_on_deep_pacman),  # take too long to evaluate
    }
    methods = ['episodic_semi_gradient', 'deep_q_learning']

    results = {}

    for env_name, block in envs.items():
        callable_env, estimator = block
        results[env_name] = {}
        for method in methods:
            q = tf.keras.models.load_model(f'{WEIGHT_PATH}{env_name}_{method}_dqn')
            results[env_name][method] = estimator(q, callable_env)

    print("Evaluation on Deep Reinforcement")
    print(results)
    print()


def compare_convergence(envs, methods, group, steps=50):
    for env in envs:
        name = f'{SLIDE_PATH}{env}_{group}_comparison'
        fig, ax = plt.subplots()
        for method in methods:
            y = load_from_pickle(f'{PLOT_PATH}reward_{env}_{method}')
            x = [i * steps for i in range(len(y))]
            ax.plot(x, y, label=f"{method}")
        ax.set_xlabel('Steps')
        ax.set_ylabel('Cumulative Reward')
        ax.set_title(f"{env.capitalize()}")
        ax.legend()
        plt.savefig(name + '.png')
        plt.show()


def generate_slide_image():
    ovr = {
        'mc': {
            'envs': ["tictactoe", "env2"],
            'methods': ['monte_carlo_es', 'monte_carlo_off_pvc', 'monte_carlo_on_pvc']
        },
        'td': {
            'envs': ["tictactoe", "env3"],
            'methods': ['sarsa', 'q_learning', 'expected_sarsa']
        },
    }

    for group, block in ovr.items():
        compare_convergence(block['envs'], block['methods'], group)
