import numpy as np

from .helper import RewardTracker, WEIGHT_PATH, save_to_pickle
from ..do_not_touch.contracts import SingleAgentEnv
from ..do_not_touch.result_structures import PolicyAndActionValueFunction


def sarsa(env: SingleAgentEnv, alpha: float, epsilon: float, gamma: float, max_episodes: int, env_name: str = ''):
    # *********** Helper *********
    name = f'{env_name}_sarsa'
    print_every_n_episodes = 10
    reward_tracker = RewardTracker(name, 5*print_every_n_episodes)
    # *********** Helper *********

    pi = {}
    b = {}
    q = {}

    for episode_id in range(max_episodes):
        env.reset()

        while not env.is_game_over():
            s = env.state_id()
            available_actions = env.available_actions_ids()

            available_actions_count = len(available_actions)

            if s not in pi:
                pi[s] = {}
                b[s] = {}
                q[s] = {}

                for a in available_actions:
                    pi[s][a] = 1.0 / available_actions_count
                    b[s][a] = 1.0 / available_actions_count
                    q[s][a] = 0.0

            optimal_a = list(q[s].keys())[np.argmax(list(q[s].values()))]

            for a_key, q_s_a in q[s].items():
                if a_key == optimal_a:
                    b[s][a_key] = 1.0 - epsilon + (epsilon / available_actions_count)
                else:
                    b[s][a_key] = epsilon / available_actions_count

            chosen_action = np.random.choice(list(b[s].keys()), 1, False, p=list(b[s].values()))[0]

            previous_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - previous_score

            s_p = env.state_id()
            next_available_actions = env.available_actions_ids()
            next_available_actions_count = len(next_available_actions)

            if env.is_game_over():
                q[s][chosen_action] += alpha * (r + 0.0 - q[s][chosen_action])
            else:
                if s_p not in pi:
                    pi[s_p] = {}
                    b[s_p] = {}
                    q[s_p] = {}

                    for a in next_available_actions:
                        pi[s_p][a] = 1.0 / next_available_actions_count
                        b[s_p][a] = 1.0 / next_available_actions_count
                        q[s_p][a] = 0.0

                next_optimal_a = list(q[s].keys())[np.argmax(list(q[s].values()))]

                for a_key, q_s_a in q[s_p].items():
                    if a_key == next_optimal_a:
                        b[s_p][a_key] = 1.0 - epsilon + (epsilon / next_available_actions_count)
                    else:
                        b[s_p][a_key] = epsilon / next_available_actions_count

                next_chosen_action = np.random.choice(list(b[s_p].keys()), 1, False, p=list(b[s_p].values()))[0]

                q[s][chosen_action] += alpha * (r + gamma * q[s_p][next_chosen_action] - q[s][chosen_action])

        # *********** Helper *********
        reward_tracker(env.score())
        # *********** Helper *********

    for s in q.keys():
        optimal_a = list(q[s].keys())[np.argmax(list(q[s].values()))]
        for a_key, q_s_a in q[s].items():
            if a_key == optimal_a:
                pi[s][a_key] = 1.0
            else:
                pi[s][a_key] = 0.0

    pi_q = PolicyAndActionValueFunction(pi, q)
    weight = f'{WEIGHT_PATH}{env_name}_sarsa_{pi_q.__class__.__name__}'
    save_to_pickle(pi_q, weight)

    reward_tracker.save_to_image()
    reward_tracker.save_to_file()

    return pi_q


def q_learning(env: SingleAgentEnv, alpha: float, epsilon: float, gamma: float, max_episodes: int, env_name: str = '') -> PolicyAndActionValueFunction:
    # *********** Helper *********
    name = f'{env_name}_q_learning'
    print_every_n_episodes = 10
    reward_tracker = RewardTracker(name, 5 * print_every_n_episodes)
    # *********** Helper *********

    pi = {}  # learned greedy policy
    b = {}  # behaviour epsilon-greedy policy
    q = {}  # action-value function of pi

    for episode_id in range(max_episodes):
        env.reset()

        while not env.is_game_over():
            s = env.state_id()
            available_actions = env.available_actions_ids()

            available_actions_count = len(available_actions)

            if s not in pi:
                pi[s] = {}
                b[s] = {}
                q[s] = {}

                for a in available_actions:
                    pi[s][a] = 1.0 / available_actions_count
                    b[s][a] = 1.0 / available_actions_count
                    q[s][a] = 0.0

            optimal_a = list(q[s].keys())[np.argmax(list(q[s].values()))]

            for a_key, q_s_a in q[s].items():
                if a_key == optimal_a:
                    b[s][a_key] = 1.0 - epsilon + (epsilon / available_actions_count)
                else:
                    b[s][a_key] = epsilon / available_actions_count

            chosen_action = np.random.choice(list(b[s].keys()), 1, False, p=list(b[s].values()))[0]

            previous_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - previous_score

            s_p = env.state_id()
            next_available_actions = env.available_actions_ids()
            next_available_actions_count = len(next_available_actions)

            if env.is_game_over():
                q[s][chosen_action] += alpha * (r + 0.0 - q[s][chosen_action])
            else:
                if s_p not in pi:
                    pi[s_p] = {}
                    b[s_p] = {}
                    q[s_p] = {}

                    for a in next_available_actions:
                        pi[s_p][a] = 1.0 / next_available_actions_count
                        b[s_p][a] = 1.0 / next_available_actions_count
                        q[s_p][a] = 0.0

                q[s][chosen_action] += alpha * (r + gamma * np.max(list(q[s_p].values())) - q[s][chosen_action])

        # *********** Helper *********
        reward_tracker(env.score())
        # *********** Helper *********

    for s in q.keys():
        optimal_a = list(q[s].keys())[np.argmax(list(q[s].values()))]
        for a_key, q_s_a in q[s].items():
            if a_key == optimal_a:
                pi[s][a_key] = 1.0
            else:
                pi[s][a_key] = 0.0

    pi_q = PolicyAndActionValueFunction(pi, q)
    weight = f'{WEIGHT_PATH}{env_name}_q_learning_{pi_q.__class__.__name__}'
    save_to_pickle(pi_q, weight)

    reward_tracker.save_to_image()
    reward_tracker.save_to_file()

    return pi_q


def expected_sarsa(env: SingleAgentEnv, alpha: float, epsilon: float, gamma: float, max_episodes: int, env_name: str = '') -> PolicyAndActionValueFunction:
    # *********** Helper *********
    name = f'{env_name}_expected_sarsa'
    print_every_n_episodes = 10
    reward_tracker = RewardTracker(name, 5*print_every_n_episodes)
    # *********** Helper *********

    pi = {}  # learned greedy policy
    b = {}  # behaviour epsilon-greedy policy
    q = {}  # action-value function of pi

    for episode_id in range(max_episodes):
        env.reset()

        while not env.is_game_over():
            s = env.state_id()
            available_actions = env.available_actions_ids()

            available_actions_count = len(available_actions)

            if s not in pi:
                pi[s] = {}
                b[s] = {}
                q[s] = {}

                for a in available_actions:
                    pi[s][a] = 1.0 / available_actions_count
                    b[s][a] = 1.0 / available_actions_count
                    q[s][a] = 0.0

            optimal_a = list(q[s].keys())[np.argmax(list(q[s].values()))]

            for a_key, q_s_a in q[s].items():
                if a_key == optimal_a:
                    b[s][a_key] = 1.0 - epsilon + (epsilon / available_actions_count)
                else:
                    b[s][a_key] = epsilon / available_actions_count

            chosen_action = np.random.choice(list(b[s].keys()), 1, False, p=list(b[s].values()))[0]

            previous_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - previous_score

            s_p = env.state_id()
            next_available_actions = env.available_actions_ids()
            next_available_actions_count = len(next_available_actions)

            if env.is_game_over():
                q[s][chosen_action] += alpha * (r + 0.0 - q[s][chosen_action])
            else:
                if s_p not in pi:
                    pi[s_p] = {}
                    b[s_p] = {}
                    q[s_p] = {}

                    for a in next_available_actions:
                        pi[s_p][a] = 1.0 / next_available_actions_count
                        b[s_p][a] = 1.0 / next_available_actions_count
                        q[s_p][a] = 0.0

                expected_value = 0
                for a_key, q_s_a in q[s_p].items():
                    expected_value += pi[s_p][a_key] * q[s_p][a_key]

                q[s][chosen_action] += alpha * (r + gamma * expected_value - q[s][chosen_action])

        # *********** Helper *********
        reward_tracker(env.score())
        # *********** Helper *********

    for s in q.keys():
        optimal_a = list(q[s].keys())[np.argmax(list(q[s].values()))]
        for a_key, q_s_a in q[s].items():
            if a_key == optimal_a:
                pi[s][a_key] = 1.0
            else:
                pi[s][a_key] = 0.0

    pi_q = PolicyAndActionValueFunction(pi, q)
    weight = f'{WEIGHT_PATH}{env_name}_expected_sarsa_{pi_q.__class__.__name__}'
    save_to_pickle(pi_q, weight)

    reward_tracker.save_to_image()
    reward_tracker.save_to_file()

    return pi_q
