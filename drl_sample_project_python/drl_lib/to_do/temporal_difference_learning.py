from monte_carlo_methods_env import TicTacToe
from temporal_difference_learning_algos import *
from ..do_not_touch.single_agent_env_wrapper import Env3


def sarsa_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = TicTacToe()

    return sarsa(env, alpha=0.1, epsilon=1, gamma=0.9, max_episodes=10000)


def q_learning_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = TicTacToe()

    return q_learning(env, alpha=0.1, epsilon=0.1, gamma=0.1, max_episodes=10000)


def expected_sarsa_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = TicTacToe()

    return expected_sarsa(env, alpha=0.1, epsilon=0.1, gamma=0.1, max_episodes=10000)


def sarsa_on_secret_env3() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()

    return sarsa(env, alpha=0.1, epsilon=0.1, gamma=0.1, max_episodes=10000)


def q_learning_on_secret_env3() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()

    return q_learning(env, alpha=0.1, epsilon=0.1, gamma=0.1, max_episodes=10000)


def expected_sarsa_on_secret_env3() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()

    return expected_sarsa(env, alpha=0.1, epsilon=0.1, gamma=0.1, max_episodes=10000)


def demo():
    print(sarsa_on_tic_tac_toe_solo())
    print(q_learning_on_tic_tac_toe_solo())
    print(expected_sarsa_on_tic_tac_toe_solo())

    print(sarsa_on_secret_env3())
    print(q_learning_on_secret_env3())
    print(expected_sarsa_on_secret_env3())
