import drl_lib.to_do.dynamic_programming as dynamic_programming
import drl_lib.to_do.monte_carlo_methods as monte_carlo_methods
import drl_lib.to_do.temporal_difference_learning as temporal_difference_learning
import drl_lib.to_do.deep_reinforcement_learning as deep_reinforcement_learning
import drl_lib.to_do.policy_gradient_methods as policy_gradient_methods
from drl_sample_project_python.drl_lib.to_play.dp_gui import *
from drl_sample_project_python.drl_lib.to_do.helper import *
from drl_sample_project_python.drl_lib.to_play.eval_envs_utilities import eval_on_dp_envs, eval_on_mc_envs, \
    eval_on_td_envs, \
    eval_on_drl_envs, generate_slide_image

if __name__ == "__main__":
    ##*********************************************##
    ##*********************************************##
    ##***************    TRAIN   ******************##
    ##*********************************************##
    ##*********************************************##

    # dynamic_programming.demo()
    # monte_carlo_methods.demo()
    # temporal_difference_learning.demo()
    deep_reinforcement_learning.demo()
    # policy_gradient_methods.demo()

    ##*********************************************##
    ##*********************************************##
    ##***************     GUI    ******************##
    ##*********************************************##
    ##*********************************************##


    # line_env = LineWorld(7)
    # pi_v = load_from_pickle(f'{WEIGHT_PATH}lineworld_policy_iteration_PolicyAndValueFunction')
    # # pi_v = load_from_pickle(f'{WEIGHT_PATH}lineworld_value_iteration_PolicyAndValueFunction')
    # show_line_world(line_env.cells_count, pi_v.v)
    # play_line_world(line_env, pi_v, 'value_iteration')

    # grid_env = GridWorld(5)
    # pi_v = load_from_pickle(f'{WEIGHT_PATH}gridworld_policy_iteration_PolicyAndValueFunction')
    # # pi_v = load_from_pickle(f'{WEIGHT_PATH}gridworld_value_iteration_PolicyAndValueFunction')
    # show_grid_world(grid_env.grid_count, pi_v.v)
    # play_grid_world(grid_env, pi_v, 'policy_iteration')

    # ttt_env = TicTacToe(3)
    # # pi_q = load_from_pickle(f'{WEIGHT_PATH}tictactoe_monte_carlo_es_PolicyAndActionValueFunction')
    # # pi_q = load_from_pickle(f'{WEIGHT_PATH}tictactoe_monte_carlo_off_pvc_PolicyAndActionValueFunction')
    # pi_q = load_from_pickle(f'{WEIGHT_PATH}tictactoe_monte_carlo_on_pvc_PolicyAndActionValueFunction')
    # play_tictactoe(ttt_env, pi_q, 'monte_carlo_es')

    # ttt_env = TicTacToe(3)
    # # pi_q = load_from_pickle(f'{WEIGHT_PATH}tictactoe_sarsa_PolicyAndActionValueFunction')
    # # pi_q = load_from_pickle(f'{WEIGHT_PATH}tictactoe_q_learning_PolicyAndActionValueFunction')
    # pi_q = load_from_pickle(f'{WEIGHT_PATH}tictactoe_expected_sarsa_PolicyAndActionValueFunction')
    # play_tictactoe(ttt_env, pi_q, 'sarsa')

    # ttt_env = TicTacToe(3, AdversaryPlayerEnum.HUMAN)  # vs Human (command line input only)
    # # pi_q = load_from_pickle(f'{WEIGHT_PATH}tictactoe_sarsa_PolicyAndActionValueFunction')
    # # pi_q = load_from_pickle(f'{WEIGHT_PATH}tictactoe_q_learning_PolicyAndActionValueFunction')
    # pi_q = load_from_pickle(f'{WEIGHT_PATH}tictactoe_expected_sarsa_PolicyAndActionValueFunction')
    # play_tictactoe(ttt_env, pi_q, 'sarsa')

    # deep_ttt_env = DeepTicTacToe(3)
    # input_dim = deep_ttt_env.state_description_length() + deep_ttt_env.max_actions_count()
    # dqn = tf.keras.models.load_model(f'{WEIGHT_PATH}tictactoe_deep_q_learning_dqn')
    # # dqn = tf.keras.models.load_model(f'{WEIGHT_PATH}tictactoe_episodic_semi_gradient_dqn')
    # play_deep_tictactoe(deep_ttt_env, dqn, 'deep_q_learning')

    # deep_ttt_env = DeepTicTacToe(3, AdversaryPlayerEnum.HUMAN)  # vs Human (command line input only)
    # input_dim = deep_ttt_env.state_description_length() + deep_ttt_env.max_actions_count()
    # dqn = tf.keras.models.load_model(f'{WEIGHT_PATH}tictactoe_deep_q_learning_dqn')
    # # dqn = tf.keras.models.load_model(f'{WEIGHT_PATH}tictactoe_episodic_semi_gradient_dqn')
    # play_deep_tictactoe(deep_ttt_env, dqn, 'deep_q_learning')

    # deep_pac_env = PacMan()
    # # dqn = tf.keras.models.load_model(f'{WEIGHT_PATH}pacman_deep_q_learning_dqn')
    # dqn = tf.keras.models.load_model(f'{WEIGHT_PATH}pacman_episodic_semi_gradient_dqn')
    # play_deep_pacman(deep_pac_env, dqn, 'deep_q_learning')

    ##*********************************************##
    ##*********************************************##
    ##***************    STATS   ******************##
    ##*********************************************##
    ##*********************************************##

    # eval_on_dp_envs()
    # eval_on_mc_envs()
    # eval_on_td_envs()
    # eval_on_drl_envs()

    ##*********************************************##
    ##*********************************************##
    ##***************    PLOTS   ******************##
    ##*********************************************##
    ##*********************************************##

    # generate_slide_image()


