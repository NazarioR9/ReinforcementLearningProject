from ..do_not_touch.deep_single_agent_with_discrete_actions_env_wrapper import Env5


class DeepQNetwork:
    """
    Contains the weights, structure of the q_network
    """
    # TODO
    pass


def episodic_semi_gradient_sarsa_on_tic_tac_toe_solo() -> DeepQNetwork:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a episodic semi gradient sarsa Algorithm in order to find the optimal epsilon-greedy action_value function
    Returns the optimal epsilon-greedy action_value function (Q(w)(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    pass


def deep_q_learning_on_tic_tac_toe_solo() -> DeepQNetwork:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a deep q Learning (DQN) algorithm in order to find the optimal action_value function
    Returns the optimal action_value function (Q(w)(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    pass


def episodic_semi_gradient_sarsa_on_pac_man() -> DeepQNetwork:
    """
    Creates a PacMan environment
    Launches a episodic semi gradient sarsa Algorithm in order to find the optimal epsilon-greedy action_value function
    Returns the optimal epsilon-greedy action_value function (Q(w)(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    pass


def deep_q_learning_on_pac_man() -> DeepQNetwork:
    """
    Creates a PacMan environment
    Launches a deep q Learning (DQN) algorithm in order to find the optimal action_value function
    Returns the optimal action_value function (Q(w)(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    pass


def episodic_semi_gradient_sarsa_on_secret_env5() -> DeepQNetwork:
    """
    Creates a Secret Env 5 environment
    Launches a episodic semi gradient sarsa Algorithm in order to find the optimal epsilon-greedy action_value function
    Returns the optimal epsilon-greedy action_value function (Q(w)(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    _env = Env5()
    pass


def deep_q_learning_on_secret_env5() -> DeepQNetwork:
    """
    Creates a Secret Env 5 environment
    Launches a deep q Learning (DQN) algorithm in order to find the optimal action_value function
    Returns the optimal action_value function (Q(w)(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    _env = Env5()
    pass


def demo():
    print(episodic_semi_gradient_sarsa_on_tic_tac_toe_solo())
    print(deep_q_learning_on_tic_tac_toe_solo())

    print(episodic_semi_gradient_sarsa_on_pac_man())
    print(deep_q_learning_on_pac_man())

    print(episodic_semi_gradient_sarsa_on_secret_env5())
    print(deep_q_learning_on_secret_env5())
