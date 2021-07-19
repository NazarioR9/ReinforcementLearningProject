from .DeepSingleAgentEnvWithDiscreteActionsStateData import *

from .bytes_wrapper import *
from . import get_dll
from .contracts import DeepSingleAgentWithDiscreteActionsEnv


class SecretDeepSingleAgentWithDiscreteActionsEnv(DeepSingleAgentWithDiscreteActionsEnv):
    def __init__(self, env):
        self.env = env
        self.data_ptr = get_dll().get_deep_single_agent_with_discrete_actions_env_state_data(self.env)
        self.data_bytes = get_bytes(self.data_ptr)
        self.data = DeepSingleAgentEnvWithDiscreteActionsStateData.GetRootAsDeepSingleAgentEnvWithDiscreteActionsStateData(
            self.data_bytes, 0)

    def state_description(self) -> np.ndarray:
        return self.data.StateDescriptionAsNumpy()

    def state_description_length(self) -> int:
        return self.data.StateDescriptionLength()

    def max_actions_count(self) -> int:
        return self.data.MaxActionsCount()

    def is_game_over(self) -> bool:
        return self.data.IsGameOver()

    def act_with_action_id(self, action_id: int):
        get_dll().act_on_deep_single_agent_with_discrete_actions_env(self.env, action_id)
        get_dll().free_wrapped_bytes(self.data_ptr)
        self.data_ptr = get_dll().get_deep_single_agent_with_discrete_actions_env_state_data(self.env)
        self.data_bytes = get_bytes(self.data_ptr)
        self.data = DeepSingleAgentEnvWithDiscreteActionsStateData.GetRootAsDeepSingleAgentEnvWithDiscreteActionsStateData(
            self.data_bytes, 0)

    def score(self) -> float:
        return self.data.Score()

    def available_actions_ids(self) -> np.ndarray:
        return self.data.AvailableActionsIdsAsNumpy()

    def reset(self):
        get_dll().reset_deep_single_agent_with_discrete_actions_env(self.env)
        get_dll().free_wrapped_bytes(self.data_ptr)
        self.data_ptr = get_dll().get_deep_single_agent_with_discrete_actions_env_state_data(self.env)
        self.data_bytes = get_bytes(self.data_ptr)
        self.data = DeepSingleAgentEnvWithDiscreteActionsStateData.GetRootAsDeepSingleAgentEnvWithDiscreteActionsStateData(
            self.data_bytes, 0)

    def view(self):
        print("It's secret !")

    def __del__(self):
        get_dll().free_wrapped_bytes(self.data_ptr)
        get_dll().delete_deep_single_agent_with_discrete_actions_env(self.env)


class Env5(SecretDeepSingleAgentWithDiscreteActionsEnv):
    def __init__(self):
        super().__init__(get_dll().create_secret_env5())
