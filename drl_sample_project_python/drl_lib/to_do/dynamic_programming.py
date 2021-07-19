from .dynamic_programming_algos import *
from .dynamic_programming_env import *
from .dynamic_programming_helper import show_grid_world
from ..do_not_touch.mdp_env_wrapper import Env1


def policy_evaluation_on_line_world() -> ValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    env = LineWorld(7)

    return policy_evaluation(env, gamma=1.0, theta=0.001)


def policy_iteration_on_line_world() -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = LineWorld(7)

    return policy_iteration(env, gamma=0.99999, theta=0.000001)


def value_iteration_on_line_world() -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = LineWorld(7)

    return value_iteration(env, gamma=0.999, theta=0.0001)


def policy_evaluation_on_grid_world() -> ValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    env = GridWorld(5)

    return policy_evaluation(env, gamma=1.0, theta=0.0001)


def policy_iteration_on_grid_world() -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = GridWorld(5)

    return policy_iteration(env, gamma=0.99999, theta=0.000001)


def value_iteration_on_grid_world() -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = GridWorld(5)

    return value_iteration(env, gamma=0.999, theta=0.0001)


def policy_evaluation_on_secret_env1() -> ValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    env = Env1()

    return policy_evaluation(env, gamma=1.0, theta=0.0001)


def policy_iteration_on_secret_env1() -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = Env1()

    return policy_iteration(env, gamma=0.99999, theta=0.000001)


def value_iteration_on_secret_env1() -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Prints the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = Env1()

    return value_iteration(env, gamma=0.999, theta=0.0001)


def demo():
    print(policy_evaluation_on_line_world())
    print(policy_iteration_on_line_world())
    print(value_iteration_on_line_world())

    print(policy_evaluation_on_grid_world())
    print(policy_iteration_on_grid_world())
    print(value_iteration_on_grid_world())

    print(policy_evaluation_on_secret_env1())
    print(policy_iteration_on_secret_env1())
    print(value_iteration_on_secret_env1())


def visualize_value_functions():
    v_pe = policy_evaluation_on_grid_world()
    pi = policy_iteration_on_grid_world()
    vi = value_iteration_on_grid_world()
    print(v_pe)
    print(pi)
    print(vi)
    print("Visu")
    print(show_grid_world(5, v_pe))
    print()
    print(show_grid_world(5, pi.v))
    print()
    print(show_grid_world(5, vi.v))
    print()
