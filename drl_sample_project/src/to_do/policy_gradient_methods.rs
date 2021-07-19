use crate::do_not_touch::secret_deep_single_agent_env_with_discrete_actions_wrapper::Env5;

/// Contains the weights, structure of the pi_network
/// todo
///
#[derive(Debug)]
struct DeepPiNetwork {
}

/// Contains the weights, structure of the v_network
/// todo
///
#[derive(Debug)]
struct DeepVNetwork {
}

/// Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
/// Launches a REINFORCE Algorithm in order to find the optimal policy
/// Returns the optimal policy network (Pi(w)(s,a))
/// Experiment with different values of hyper parameters and choose the most appropriate combination
fn reinforce_on_tic_tac_toe_solo() -> DeepPiNetwork {
    todo!()
}

/// Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
/// Launches a REINFORCE with Baseline algorithm  in order to find the optimal policy and its value function
/// Returns the optimal policy network (Pi(w)(s,a)) and its value function (V(w)(s))
/// Experiment with different values of hyper parameters and choose the most appropriate combination
fn reinforce_with_baseline_on_tic_tac_toe_solo() -> (DeepPiNetwork, DeepVNetwork) {
    todo!()
}

/// Creates a PacMan environment
/// Launches a REINFORCE Algorithm in order to find the optimal policy
/// Returns the optimal policy network (Pi(w)(s,a))
/// Experiment with different values of hyper parameters and choose the most appropriate combination
fn reinforce_on_pac_man() -> DeepPiNetwork {
    todo!()
}

/// Creates a PacMan environment
/// Launches a REINFORCE with Baseline algorithm in order to find the optimal policy and its value function
/// Returns the optimal policy network (Pi(w)(s,a)) and its value function (V(w)(s))
/// Experiment with different values of hyper parameters and choose the most appropriate combination
fn reinforce_with_baseline_on_pac_man() -> (DeepPiNetwork, DeepVNetwork) {
    todo!()
}

/// Creates a Secret Env 5 environment
/// Launches a REINFORCE Algorithm in order to find the optimal policy
/// Returns the optimal policy network (Pi(w)(s,a))
/// Experiment with different values of hyper parameters and choose the most appropriate combination
fn reinforce_on_secret_env_5() -> DeepPiNetwork {
    let _env = Env5::new();
    todo!()
}

/// Creates a Secret Env 5 environment
/// Launches a REINFORCE with Baseline algorithm  in order to find the optimal policy and its value function
/// Returns the optimal policy network (Pi(w)(s,a)) and its value function (V(w)(s))
/// Experiment with different values of hyper parameters and choose the most appropriate combination
fn reinforce_with_baseline_on_secret_env_5() -> (DeepPiNetwork, DeepVNetwork) {
    let _env = Env5::new();
    todo!()
}


pub fn demo() {
    dbg!(reinforce_on_tic_tac_toe_solo());
    dbg!(reinforce_with_baseline_on_tic_tac_toe_solo());

    dbg!(reinforce_on_pac_man());
    dbg!(reinforce_with_baseline_on_pac_man());

    dbg!(reinforce_on_secret_env_5());
    dbg!(reinforce_with_baseline_on_secret_env_5());
}