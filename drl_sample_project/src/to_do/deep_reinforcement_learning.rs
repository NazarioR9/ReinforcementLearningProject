use crate::do_not_touch::secret_deep_single_agent_env_with_discrete_actions_wrapper::Env5;

/// Contains the weights, structure of the q_network
/// todo
///
#[derive(Debug)]
struct DeepQNetwork {
}

/// Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
/// Launches a episodic semi gradient sarsa Algorithm in order to find the optimal epsilon-greedy action_value function
/// Returns the optimal epsilon-greedy action_value function (Q(w)(s,a))
/// Experiment with different values of hyper parameters and choose the most appropriate combination
fn episodic_semi_gradient_sarsa_on_tic_tac_toe_solo() -> DeepQNetwork {
    todo!()
}

/// Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
/// Launches a deep q Learning (DQN) algorithm in order to find the optimal action_value function
/// Returns the optimal action_value function (Q(w)(s,a))
/// Experiment with different values of hyper parameters and choose the most appropriate combination
fn deep_q_learning_on_tic_tac_toe_solo() -> DeepQNetwork {
    todo!()
}

/// Creates a PacMan environment
/// Launches a episodic semi gradient sarsa Algorithm in order to find the optimal epsilon-greedy action_value function
/// Returns the optimal epsilon-greedy action_value function (Q(w)(s,a))
/// Experiment with different values of hyper parameters and choose the most appropriate combination
fn episodic_semi_gradient_sarsa_on_pac_man() -> DeepQNetwork {
    todo!()
}

/// Creates a PacMan environment
/// Launches a deep q Learning (DQN) algorithm in order to find the optimal action_value function
/// Returns the optimal action_value function (Q(w)(s,a))
/// Experiment with different values of hyper parameters and choose the most appropriate combination
fn deep_q_learning_on_pac_man() -> DeepQNetwork {
    todo!()
}

/// Creates a Secret Env 5 environment
/// Launches a episodic semi gradient sarsa Algorithm in order to find the optimal epsilon-greedy action_value function
/// Returns the optimal epsilon-greedy action_value function (Q(w)(s,a))
/// Experiment with different values of hyper parameters and choose the most appropriate combination
fn episodic_semi_gradient_sarsa_on_secret_env5() -> DeepQNetwork {
    let _env = Env5::new();
    todo!()
}

/// Creates a Secret Env 5 environment
/// Launches a deep q Learning (DQN) algorithm in order to find the optimal action_value function
/// Returns the optimal action_value function (Q(w)(s,a))
/// Experiment with different values of hyper parameters and choose the most appropriate combination
fn deep_q_learning_on_secret_env5() -> DeepQNetwork {
    let _env = Env5::new();
    todo!()
}


pub fn demo() {
    dbg!(episodic_semi_gradient_sarsa_on_tic_tac_toe_solo());
    dbg!(deep_q_learning_on_tic_tac_toe_solo());

    dbg!(episodic_semi_gradient_sarsa_on_pac_man());
    dbg!(deep_q_learning_on_pac_man());

    dbg!(episodic_semi_gradient_sarsa_on_secret_env5());
    dbg!(deep_q_learning_on_secret_env5());
}