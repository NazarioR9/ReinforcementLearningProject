use drl_contracts::contracts::{DeepSingleAgentEnvWithDiscreteActions};

use crate::do_not_touch::bytes_wrapper::{free_wrapped_bytes, get_bytes, WrappedData};
use crate::do_not_touch::secret_envs_dynamic_libs_wrapper::SecretEnvDynamicLibWrapper;
use crate::do_not_touch::deep_single_agent_env_with_discrete_actions_state_data_generated::{root_as_deep_single_agent_env_with_discrete_actions_state_data, DeepSingleAgentEnvWithDiscreteActionsStateData};

struct SecretDeepSingleAgentWithDiscreteActionsEnv<'a> {
    env: *mut std::ffi::c_void,
    wrapper: SecretEnvDynamicLibWrapper,
    data_ptr: *mut WrappedData,
    data: DeepSingleAgentEnvWithDiscreteActionsStateData<'a>,
}

impl<'a> SecretDeepSingleAgentWithDiscreteActionsEnv<'a> {
    pub fn new(env: *mut std::ffi::c_void) -> Self {
        let wrapper = SecretEnvDynamicLibWrapper::new();
        unsafe {
            let data_ptr = wrapper.get_deep_single_agent_with_discrete_actions_env_state_data()(env);
            let data_bytes = get_bytes(data_ptr);
            let data = root_as_deep_single_agent_env_with_discrete_actions_state_data(data_bytes).unwrap();

            SecretDeepSingleAgentWithDiscreteActionsEnv {
                env,
                wrapper,
                data_ptr,
                data,
            }
        }
    }
}

impl<'a> Drop for SecretDeepSingleAgentWithDiscreteActionsEnv<'a> {
    fn drop(&mut self) {
        free_wrapped_bytes(self.data_ptr);
        unsafe {
            self.wrapper.delete_deep_single_agent_with_discrete_actions_env()(self.env);
        }
    }
}

impl<'a> DeepSingleAgentEnvWithDiscreteActions for SecretDeepSingleAgentWithDiscreteActionsEnv<'a> {
    fn state_description_length(&self) -> usize {
        self.data.state_description_size() as usize
    }

    fn state_description(&self) -> Vec<f32> {
        self.data.state_description().unwrap().iter().collect()
    }

    fn max_actions_count(&self) -> usize {
        self.data.max_actions_count() as usize
    }

    fn is_game_over(&self) -> bool {
        self.data.is_game_over()
    }

    fn act_with_action_id(&mut self, action_id: usize) {
        unsafe { self.wrapper.act_on_deep_single_agent_with_discrete_actions_env()(self.env, action_id) };
        free_wrapped_bytes(self.data_ptr);
        unsafe {
            self.data_ptr = self.wrapper.get_deep_single_agent_with_discrete_actions_env_state_data()(self.env);
            let data_bytes = get_bytes(self.data_ptr);
            self.data = root_as_deep_single_agent_env_with_discrete_actions_state_data(data_bytes).unwrap();
        }
    }

    fn score(&self) -> f32 {
        self.data.score()
    }

    fn available_actions_ids(&self) -> Vec<usize> {
        self.data.available_actions_ids().unwrap().iter().map(|a| a as usize).collect::<Vec<_>>()
    }

    fn reset(&mut self) {
        unsafe { self.wrapper.reset_deep_single_agent_with_discrete_actions_env()(self.env) }
        free_wrapped_bytes(self.data_ptr);
        unsafe {
            self.data_ptr = self.wrapper.get_deep_single_agent_with_discrete_actions_env_state_data()(self.env);
            let data_bytes = get_bytes(self.data_ptr);
            self.data = root_as_deep_single_agent_env_with_discrete_actions_state_data(data_bytes).unwrap();
        }
    }

    fn view(&self) {
        println!("It's secret !")
    }
}

pub struct Env5<'a> {
    secret_env: SecretDeepSingleAgentWithDiscreteActionsEnv<'a>,
}

impl<'a> Env5<'a> {
    pub fn new() -> Self {
        unsafe {
            Env5 {
                secret_env: SecretDeepSingleAgentWithDiscreteActionsEnv::new(SecretEnvDynamicLibWrapper::new().create_secret_env5()())
            }
        }
    }
}

impl<'a> DeepSingleAgentEnvWithDiscreteActions for Env5<'a> {
    fn state_description_length(&self) -> usize {
        self.secret_env.state_description_length()
    }

    fn state_description(&self) -> Vec<f32> {
        self.secret_env.state_description()
    }

    fn max_actions_count(&self) -> usize {
        self.secret_env.max_actions_count()
    }

    fn is_game_over(&self) -> bool {
        self.secret_env.is_game_over()
    }

    fn act_with_action_id(&mut self, action_id: usize) {
        self.secret_env.act_with_action_id(action_id)
    }

    fn score(&self) -> f32 {
        self.secret_env.score()
    }

    fn available_actions_ids(&self) -> Vec<usize> {
        self.secret_env.available_actions_ids()
    }

    fn reset(&mut self) {
        self.secret_env.reset()
    }

    fn view(&self) {
        self.secret_env.view()
    }
}