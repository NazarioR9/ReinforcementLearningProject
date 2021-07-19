use crate::to_do::{dynamic_programming, monte_carlo_methods, temporal_difference_learning, deep_reinforcement_learning, policy_gradient_methods};
use crate::do_not_touch::secret_mdp_env_wrapper::Env1;
use drl_contracts::contracts::{MDPEnv, SingleAgentEnv, DeepSingleAgentEnvWithDiscreteActions};
use crate::do_not_touch::secret_single_agent_env_wrapper::{Env2, Env3};
use crate::do_not_touch::secret_deep_single_agent_env_with_discrete_actions_wrapper::Env5;

pub mod to_do;
pub mod do_not_touch;


fn main() {
    let env = Env1::new();
    dbg!(env.actions());
    dbg!(env.rewards());
    dbg!(env.states());
    dbg!(env.is_state_terminal(0));
    dbg!(env.transition_probability(0, 0, 0, 0f32));
    dbg!(env.view_state(0));

    let mut env = Env2::new();
    dbg!(env.reset());
    dbg!(env.available_actions_ids());
    dbg!(env.score());
    dbg!(env.act_with_action_id(0));
    dbg!(env.available_actions_ids());
    dbg!(env.state_id());
    dbg!(env.reset_random());

    let mut env = Env3::new();
    dbg!(env.reset());
    dbg!(env.available_actions_ids());
    dbg!(env.score());
    dbg!(env.act_with_action_id(0));
    dbg!(env.available_actions_ids());
    dbg!(env.state_id());
    dbg!(env.reset_random());

    let mut env = Env5::new();
    dbg!(env.reset());
    dbg!(env.available_actions_ids());
    dbg!(env.score());
    dbg!(env.act_with_action_id(0));
    dbg!(env.available_actions_ids());
    dbg!(env.state_description());
    dbg!(env.state_description_length());
    dbg!(env.max_actions_count());


    dynamic_programming::demo();
    monte_carlo_methods::demo();
    temporal_difference_learning::demo();
    deep_reinforcement_learning::demo();
    policy_gradient_methods::demo();
}
