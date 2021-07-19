from ..do_not_touch.deep_single_agent_with_discrete_actions_env_wrapper import Env5


class DeepPiNetwork:
    """
    Contains the weights, structure of the pi_network (policy)
    """
    # TODO


class DeepVNetwork:
    """
    Contains the weights, structure of the v_network (baseline)
    """
    # TODO
    pass


def reinforce_on_tic_tac_toe_solo() -> DeepPiNetwork:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a REINFORCE Algorithm in order to find the optimal policy
    Returns the optimal policy network (Pi(w)(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    pass


def reinforce_with_baseline_on_tic_tac_toe_solo() -> (DeepPiNetwork, DeepVNetwork):
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a REINFORCE with Baseline algorithm  in order to find the optimal policy and its value function
    Returns the optimal policy network (Pi(w)(s,a)) and its value function (V(w)(s))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    pass


def reinforce_on_pac_man() -> DeepPiNetwork:
    """
    Creates a PacMan environment
    Launches a REINFORCE Algorithm in order to find the optimal policy
    Returns the optimal policy network (Pi(w)(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    pass


def reinforce_with_baseline_on_pac_man() -> (DeepPiNetwork, DeepVNetwork):
    """
    Creates a PacMan environment
    Launches a REINFORCE with Baseline algorithm in order to find the optimal policy and its value function
    Returns the optimal policy network (Pi(w)(s,a)) and its value function (V(w)(s))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    pass


def reinforce_on_secret_env_5() -> DeepPiNetwork:
    """
    Creates a Secret Env 5 environment
    Launches a REINFORCE Algorithm in order to find the optimal policy
    Returns the optimal policy network (Pi(w)(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    _env = Env5()
    pass


def reinforce_with_baseline_on_secret_env_5() -> (DeepPiNetwork, DeepVNetwork):
    """
    Creates a Secret Env 5 environment
    Launches a REINFORCE with Baseline algorithm  in order to find the optimal policy and its value function
    Returns the optimal policy network (Pi(w)(s,a)) and its value function (V(w)(s))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    _env = Env5()
    pass


def demo():
    print(reinforce_on_tic_tac_toe_solo())
    print(reinforce_with_baseline_on_tic_tac_toe_solo())

    print(reinforce_on_pac_man())
    print(reinforce_with_baseline_on_pac_man())

    print(reinforce_on_secret_env_5())
    print(reinforce_with_baseline_on_secret_env_5())
