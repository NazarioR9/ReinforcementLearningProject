from .deep_reinforcement_learning_algos import DeepQNetwork, episodic_semi_gradient_sarsa, deep_q_learning
from .envs import *
from ..do_not_touch.deep_single_agent_with_discrete_actions_env_wrapper import Env5


def episodic_semi_gradient_sarsa_on_tic_tac_toe_solo() -> DeepQNetwork:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a episodic semi gradient sarsa Algorithm in order to find the optimal epsilon-greedy action_value function
    Returns the optimal epsilon-greedy action_value function (Q(w)(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = DeepTicTacToe()

    return episodic_semi_gradient_sarsa(env, epsilon=0.1, gamma=0.9, max_episodes_count=3000, env_name='tictactoe')


def deep_q_learning_on_tic_tac_toe_solo() -> DeepQNetwork:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a deep q Learning (DQN) algorithm in order to find the optimal action_value function
    Returns the optimal action_value function (Q(w)(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = DeepTicTacToe()

    return deep_q_learning(env, epsilon=0.1, gamma=0.9, max_episodes_count=3000, env_name='tictactoe')


def episodic_semi_gradient_sarsa_on_pac_man() -> DeepQNetwork:
    """
    Creates a PacMan environment
    Launches a episodic semi gradient sarsa Algorithm in order to find the optimal epsilon-greedy action_value function
    Returns the optimal epsilon-greedy action_value function (Q(w)(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = PacMan()

    return episodic_semi_gradient_sarsa(env, epsilon=0.1, gamma=0.9, max_episodes_count=3000, env_name='pacman')


def deep_q_learning_on_pac_man() -> DeepQNetwork:
    """
    Creates a PacMan environment
    Launches a deep q Learning (DQN) algorithm in order to find the optimal action_value function
    Returns the optimal action_value function (Q(w)(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = PacMan()

    return deep_q_learning(env, epsilon=0.1, gamma=0.9, max_episodes_count=3000, env_name='pacman')


def episodic_semi_gradient_sarsa_on_secret_env5() -> DeepQNetwork:
    """
    Creates a Secret Env 5 environment
    Launches a episodic semi gradient sarsa Algorithm in order to find the optimal epsilon-greedy action_value function
    Returns the optimal epsilon-greedy action_value function (Q(w)(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    _env = Env5()

    return episodic_semi_gradient_sarsa(_env, epsilon=0.1, gamma=0.9, max_episodes_count=1000, env_name='env5')


def deep_q_learning_on_secret_env5() -> DeepQNetwork:
    """
    Creates a Secret Env 5 environment
    Launches a deep q Learning (DQN) algorithm in order to find the optimal action_value function
    Returns the optimal action_value function (Q(w)(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    _env = Env5()

    return deep_q_learning(_env, epsilon=0.1, gamma=0.9, max_episodes_count=1000, env_name='env5')


def demo():
    # print(episodic_semi_gradient_sarsa_on_tic_tac_toe_solo())
    # print(deep_q_learning_on_tic_tac_toe_solo())
    #
    print(episodic_semi_gradient_sarsa_on_pac_man())
    # print(deep_q_learning_on_pac_man())
    #
    # print(episodic_semi_gradient_sarsa_on_secret_env5())
    # print(deep_q_learning_on_secret_env5())
