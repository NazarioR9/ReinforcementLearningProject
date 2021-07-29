import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, activations
from tqdm.auto import tqdm

from .helper import WEIGHT_PATH, RewardTracker
from ..do_not_touch.contracts import DeepSingleAgentWithDiscreteActionsEnv

# tf.config.set_visible_devices([], 'GPU')

MIN_REPLAY_MEMORY = 50
MINI_BATCH_SIZE = 50
UPDATE_TARGET_STEP = 5


class DeepQNetwork(tf.keras.Model):
    def __init__(self, input_dim: int, dense: list = [16]):
        super(DeepQNetwork, self).__init__()
        self.dense_layers = tf.keras.Sequential()
        self.dense_layers.add(layers.Dense(dense[0], activation=activations.tanh, input_dim=input_dim))

        for dim in dense[1:]:
            self.dense_layers.add(layers.Dense(dim, activation=activations.tanh))

        self.dense_layers.add(layers.Dense(1, activation=activations.linear))

    def call(self, inputs, training=None, mask=None):
        return self.dense_layers(inputs)


class DQL_DQN:
    def __init__(self, state_description_length: int, max_actions_count: int, dense: list = [16]):

        input_dim = state_description_length + max_actions_count

        self.max_actions_count = max_actions_count
        self.main_model = DeepQNetwork(input_dim, dense)  # for training
        self.target_model = DeepQNetwork(input_dim, dense)  # for for predict
        self.update_target_model()

        self.replay_memory = []  # s, a, r, s_p
        self.steps = 0

    def compile(self, **kwargs):
        self.main_model.compile(**kwargs)
        self.target_model.compile(**kwargs)

    def train_on_batch(self, inputs, targets):
        self.main_model.train_on_batch(inputs, targets)

    def predict(self, inputs):
        return self.target_model.predict(inputs)

    def update_target_model(self):
        self.target_model.set_weights(self.main_model.get_weights())

    def save(self, weight):
        self.target_model.save(weight)

    def load_weights(self, weight):
        # self.main_model.save(weight)
        # self.target_model.save(weight)
        pass

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY:
            return

        mini_batch = random.sample(self.replay_memory, MINI_BATCH_SIZE)
        q_inputs = [np.hstack([transition[0], tf.keras.utils.to_categorical(transition[1], self.max_actions_count)]) for transition in mini_batch]
        q_inputs = np.vstack(q_inputs)
        targets = [transition[2] for transition in mini_batch]

        self.main_model.train_on_batch(np.array(q_inputs), np.array(targets))

        if self.steps % UPDATE_TARGET_STEP or self.steps == 0:
            self.update_target_model()

        self.steps += 1


def episodic_semi_gradient_sarsa(env: DeepSingleAgentWithDiscreteActionsEnv, epsilon: float, gamma: float, max_episodes_count: int, env_name='') -> DeepQNetwork:
    pre_warm = MIN_REPLAY_MEMORY
    print_every_n_episodes = 10

    name = f'{env_name}_episodic_semi_gradient_dqn'
    reward_tracker = RewardTracker(name, 5*print_every_n_episodes)

    state_description_length = env.state_description_length()
    max_actions_count = env.max_actions_count()

    q = DeepQNetwork(state_description_length + max_actions_count)
    q.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mse)

    for episode_id in tqdm(range(max_episodes_count)):
        env.reset()

        while not env.is_game_over():
            s = env.state_description()
            available_actions = env.available_actions_ids()

            # all_q_inputs = np.zeros((len(available_actions), state_description_length + max_actions_count))
            # for i, a in enumerate(available_actions):
            #     all_q_inputs[i] = np.hstack([s, tf.keras.utils.to_categorical(a, max_actions_count)])
            # all_q_values = np.squeeze(q.predict(all_q_inputs))
            #
            # if all_q_values.shape == ():
            #     all_q_values = [all_q_values[()]]

            chosen_action_q_value = None
            if episode_id < pre_warm or np.random.uniform(0.0, 1.0) < epsilon:
                chosen_action = np.random.choice(available_actions)
                # chosen_action_id = np.where(available_actions == chosen_action)[0][0]
                # chosen_action_q_value = all_q_values[chosen_action_id]
            else:
                all_q_inputs = np.zeros((len(available_actions), state_description_length + max_actions_count))
                for i, a in enumerate(available_actions):
                    all_q_inputs[i] = np.hstack([s, tf.keras.utils.to_categorical(a, max_actions_count)])
                all_q_values = np.squeeze(q.predict(all_q_inputs))

                chosen_action = available_actions[np.argmax(all_q_values)]
                chosen_action_q_value = np.max(all_q_values)

            if episode_id % print_every_n_episodes == 0:
                print(f'State Description : {s}')
                print(f'Chosen action : {chosen_action}')
                print(f'Chosen action value : {chosen_action_q_value}')

            previous_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - previous_score
            s_p = env.state_description()

            if env.is_game_over():
                target = r
                q_inputs = np.hstack([s, tf.keras.utils.to_categorical(chosen_action, max_actions_count)])
                q.train_on_batch(np.array([q_inputs]), np.array([target]))
                break

            next_available_actions = env.available_actions_ids()

            if episode_id < pre_warm or np.random.uniform(0.0, 1.0) < epsilon:
                next_chosen_action = np.random.choice(next_available_actions)
                next_q_inputs = np.hstack([s_p, tf.keras.utils.to_categorical(next_chosen_action, max_actions_count)])
                next_chosen_action_q_value = q.predict(np.array([next_q_inputs]))[0][0]
            else:
                next_chosen_action = None
                next_chosen_action_q_value = None
                for a in next_available_actions:
                    q_inputs = np.hstack([s_p, tf.keras.utils.to_categorical(a, max_actions_count)])
                    q_value = q.predict(np.array([q_inputs]))[0][0]
                    if next_chosen_action is None or next_chosen_action_q_value < q_value:
                        next_chosen_action = a
                        next_chosen_action_q_value = q_value

            target = r + gamma * next_chosen_action_q_value

            q_inputs = np.hstack([s, tf.keras.utils.to_categorical(chosen_action, max_actions_count)])
            q.train_on_batch(np.array([q_inputs]), np.array([target]))

        reward_tracker(env.score())

        if (episode_id+1) % 100 == 0:
            weight = f'{WEIGHT_PATH}{env_name}_episodic_semi_gradient_dqn'
            q.save(weight)

    weight = f'{WEIGHT_PATH}{env_name}_episodic_semi_gradient_dqn'
    q.save(weight)

    reward_tracker.save_to_image()
    reward_tracker.save_to_file()

    return q


def deep_q_learning(env: DeepSingleAgentWithDiscreteActionsEnv, epsilon: float, gamma: float, max_episodes_count: int, env_name: str = '') -> DeepQNetwork:
    pre_warm = MIN_REPLAY_MEMORY
    print_every_n_episodes = 10

    name = f'{env_name}_deep_q_learning_dqn'
    reward_tracker = RewardTracker(name, 5*print_every_n_episodes)

    state_description_length = env.state_description_length()
    max_actions_count = env.max_actions_count()

    dqn = DQL_DQN(state_description_length, max_actions_count)
    dqn.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mse)

    for episode_id in tqdm(range(max_episodes_count)):
        env.reset()

        while not env.is_game_over():
            s = env.state_description()
            available_actions = env.available_actions_ids()

            # all_q_inputs = np.zeros((len(available_actions), state_description_length + max_actions_count))
            # for i, a in enumerate(available_actions):
            #     all_q_inputs[i] = np.hstack([s, tf.keras.utils.to_categorical(a, max_actions_count)])
            # all_q_values = np.squeeze(q.predict(all_q_inputs))
            #
            # if all_q_values.shape == ():
            #     all_q_values = [all_q_values[()]]

            chosen_action_q_value = None
            if episode_id < pre_warm or np.random.uniform(0.0, 1.0) < epsilon:
                chosen_action = np.random.choice(available_actions)
                # chosen_action_id = np.where(available_actions == chosen_action)[0][0]
                # chosen_action_q_value = all_q_values[chosen_action_id]
            else:
                all_q_inputs = np.zeros((len(available_actions), state_description_length + max_actions_count))
                for i, a in enumerate(available_actions):
                    all_q_inputs[i] = np.hstack([s, tf.keras.utils.to_categorical(a, max_actions_count)])
                all_q_values = np.squeeze(dqn.predict(all_q_inputs))

                chosen_action = available_actions[np.argmax(all_q_values)]
                chosen_action_q_value = np.max(all_q_values)

            if episode_id % print_every_n_episodes == 0:
                print(f'State Description : {s}')
                print(f'Chosen action : {chosen_action}')
                print(f'Chosen action value : {chosen_action_q_value}')

            previous_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - previous_score
            s_p = env.state_description()

            if env.is_game_over():
                target = r
                dqn.update_replay_memory([s, chosen_action, target, s_p])
                break

            next_available_actions = env.available_actions_ids()

            if episode_id < pre_warm or np.random.uniform(0.0, 1.0) < epsilon:
                next_chosen_action = np.random.choice(next_available_actions)
                next_q_inputs = np.hstack([s_p, tf.keras.utils.to_categorical(next_chosen_action, max_actions_count)])
                next_chosen_action_q_value = dqn.predict(np.array([next_q_inputs]))[0][0]
            else:
                next_chosen_action = None
                next_chosen_action_q_value = None
                for a in next_available_actions:
                    q_inputs = np.hstack([s_p, tf.keras.utils.to_categorical(a, max_actions_count)])
                    q_value = dqn.predict(np.array([q_inputs]))[0][0]
                    if next_chosen_action is None or next_chosen_action_q_value < q_value:
                        next_chosen_action = a
                        next_chosen_action_q_value = q_value

            target = r + gamma * next_chosen_action_q_value
            dqn.update_replay_memory([s, chosen_action, target, s_p])

            dqn.train()
        reward_tracker(env.score())

        if (episode_id+1) % 100 == 0:
            weight = f'{WEIGHT_PATH}{env_name}_deep_q_learning_dqn'
            dqn.save(weight)
            reward_tracker.save_to_file()

    weight = f'{WEIGHT_PATH}{env_name}_deep_q_learning_dqn'
    dqn.save(weight)

    reward_tracker.save_to_image()
    reward_tracker.save_to_file()

    return dqn.target_model
