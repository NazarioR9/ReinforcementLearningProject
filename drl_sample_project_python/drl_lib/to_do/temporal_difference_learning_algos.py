import numpy as np

from ..do_not_touch.contracts import SingleAgentEnv
from ..do_not_touch.result_structures import PolicyAndActionValueFunction


def sarsa(env: SingleAgentEnv, alpha: float, epsilon: float, gamma: float, max_episodes: int):
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
                    b[s][a_key] = 1.0 - epsilon + epsilon / available_actions_count
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
                        b[s_p][a_key] = 1.0 - epsilon + epsilon / next_available_actions_count
                    else:
                        b[s_p][a_key] = epsilon / next_available_actions_count

                next_chosen_action = np.random.choice(list(b[s_p].keys()), 1, False, p=list(b[s_p].values()))[0]

                q[s][chosen_action] += alpha * (r + gamma * q[s_p][next_chosen_action] - q[s][chosen_action])

    for s in q.keys():
        optimal_a = list(q[s].keys())[np.argmax(list(q[s].values()))]
        for a_key, q_s_a in q[s].items():
            if a_key == optimal_a:
                pi[s][a_key] = 1.0
            else:
                pi[s][a_key] = 0.0

    return PolicyAndActionValueFunction(pi, q)


def q_learning(env: SingleAgentEnv, alpha: float, epsilon: float, gamma: float, max_episodes: int) -> PolicyAndActionValueFunction:
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
                    b[s][a_key] = 1.0 - epsilon + epsilon / available_actions_count
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

    for s in q.keys():
        optimal_a = list(q[s].keys())[np.argmax(list(q[s].values()))]
        for a_key, q_s_a in q[s].items():
            if a_key == optimal_a:
                pi[s][a_key] = 1.0
            else:
                pi[s][a_key] = 0.0

    return PolicyAndActionValueFunction(pi, q)


def expected_sarsa(env: SingleAgentEnv, alpha: float, epsilon: float, gamma: float, max_episodes: int) -> PolicyAndActionValueFunction:
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
                    b[s][a_key] = 1.0 - epsilon + epsilon / available_actions_count
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
                for a_key, q_s_a in q[s].items():
                    expected_value += pi[s_p][a_key] * q[s_p][a_key]

                q[s][chosen_action] += alpha * (r + gamma * expected_value - q[s][chosen_action])

    for s in q.keys():
        optimal_a = list(q[s].keys())[np.argmax(list(q[s].values()))]
        for a_key, q_s_a in q[s].items():
            if a_key == optimal_a:
                pi[s][a_key] = 1.0
            else:
                pi[s][a_key] = 0.0

    return PolicyAndActionValueFunction(pi, q)
