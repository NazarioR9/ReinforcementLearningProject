import random
import sys
import time
from typing import Union

import pygame
from pygame.locals import QUIT

import tensorflow as tf

from ..do_not_touch.result_structures import PolicyAndValueFunction, PolicyAndActionValueFunction
from ..to_do.deep_reinforcement_learning_algos import DeepQNetwork
from ..to_do.envs import *

SIZE = 50


class LineWorldGui(pygame.sprite.Sprite):
    def __init__(self, cell_count):
        super().__init__()
        self.WIDTH = SIZE * cell_count
        self.HEIGHT = SIZE
        self.CENTER = self.HEIGHT // 2

        self.surf = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.surf.fill((255, 255, 255))

        self.cell_count = cell_count

        self.player_pos = cell_count // 2
        self.player = None

        self.build_rects()

    def build_rects(self):
        for c in range(1, self.cell_count):
            pygame.draw.line(self.surf, (0, 0, 0), (c * self.HEIGHT, 0), (c * self.HEIGHT, self.HEIGHT))
        pygame.draw.rect(self.surf, (0, 255, 0), (self.WIDTH - self.HEIGHT, 0, self.HEIGHT, self.HEIGHT))
        pygame.draw.rect(self.surf, (255, 0, 0), (0, 0, self.HEIGHT, self.HEIGHT))

    def draw_agent(self):
        x = self.player_pos * self.HEIGHT + (self.CENTER // 2)
        self.player = pygame.draw.rect(self.surf, (0, 0, 255), (x, self.CENTER // 2, SIZE // 2, SIZE // 2))

    def move_agent(self, pos):
        self.player_pos = pos
        self.draw_agent()

    def reset(self):
        self.surf.fill((255, 255, 255))
        self.build_rects()
        self.draw_agent()


class GridWorldGui(pygame.sprite.Sprite):
    def __init__(self, cell_count):
        super().__init__()
        self.HEIGHT = self.WIDTH = SIZE * cell_count
        self.CENTER = self.HEIGHT // 2

        self.surf = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.surf.fill((255, 255, 255))

        self.cell_count = cell_count

        self.player_pos = (0, 0)
        self.player = None

        self.build_rects()

    def build_rects(self):
        pygame.draw.rect(self.surf, (255, 0, 0), (self.WIDTH - SIZE, 0, SIZE, SIZE))
        pygame.draw.rect(self.surf, (0, 255, 0), (self.WIDTH - SIZE, self.HEIGHT - SIZE, SIZE, SIZE))
        for c in range(1, self.cell_count):
            pygame.draw.line(self.surf, (0, 0, 0), (c * SIZE, 0), (c * SIZE, self.HEIGHT))
            pygame.draw.line(self.surf, (0, 0, 0), (0, c * SIZE), (self.WIDTH, c * SIZE))

    def draw_agent(self):
        x, y = self.player_pos
        x = x * SIZE + (SIZE // 4)
        y = y * SIZE + (SIZE // 4)
        self.player = pygame.draw.rect(self.surf, (0, 0, 255), (x, y, SIZE // 2, SIZE // 2))

    def move_agent(self, pos):
        self.player_pos = (pos % self.cell_count, pos // self.cell_count)
        self.draw_agent()

    def reset(self):
        self.surf.fill((255, 255, 255))
        self.build_rects()
        self.draw_agent()


class TicTacToeGui(pygame.sprite.Sprite):
    def __init__(self, grid_count):
        super().__init__()
        self.HEIGHT = self.WIDTH = SIZE * grid_count
        self.CENTER = self.HEIGHT // 2

        self.surf = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.surf.fill((255, 255, 255))

        self.cell_count = grid_count
        self.colors = [(0, 0, 255), (255, 0, 0)]

        self.build_rects()

    def build_rects(self):
        for c in range(1, self.cell_count):
            pygame.draw.line(self.surf, (0, 0, 0), (c * SIZE, 0), (c * SIZE, self.HEIGHT))
            pygame.draw.line(self.surf, (0, 0, 0), (0, c * SIZE), (self.WIDTH, c * SIZE))

    def draw(self, env: Union[TicTacToe, DeepTicTacToe]):
        for player_id, color in zip([env.agent_player_id, env.adv_player_id], self.colors):
            for pos in np.where(env.map == player_id)[0]:
                self.draw_move(color, pos)

    def draw_move(self, color, pos):
        x, y = (pos % self.cell_count, pos // self.cell_count)
        x = x * SIZE
        y = y * SIZE
        pygame.draw.rect(self.surf, color, (x, y, SIZE, SIZE))

    def reset(self):
        self.surf.fill((255, 255, 255))
        self.build_rects()


class PacManGui(pygame.sprite.Sprite):
    def __init__(self, grid_count):
        super().__init__()
        self.HEIGHT = self.WIDTH = SIZE * grid_count
        self.CENTER = self.HEIGHT // 2

        self.surf = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.surf.fill((255, 255, 255))

        self.cell_count = grid_count
        self.pacman_color = (255, 255, 0)
        self.ghost_colors = [(255, 0, 0), (255, 192, 203), (0, 0, 255), (255, 0, 255)]
        self.entity_to_color = {"food": (0, 255, 0), "wall": (0, 0, 0)}

        self.build_rects()

    def build_rects(self):
        for c in range(1, self.cell_count):
            pygame.draw.line(self.surf, (0, 0, 0), (c * SIZE, 0), (c * SIZE, self.HEIGHT))
            pygame.draw.line(self.surf, (0, 0, 0), (0, c * SIZE), (self.WIDTH, c * SIZE))

    def draw(self, env: PacMan):
        pac_map = env.map
        entity_ids = env.entity_ids

        for pos, entity in enumerate(pac_map):
            if entity == entity_ids['food']:
                color = self.entity_to_color['food']
                self.draw_pac_object(color, pos, 8)

            if entity == entity_ids['wall']:
                color = self.entity_to_color['wall']
                self.draw_wall(color, pos)

        self.draw_pac_object(self.pacman_color, env.agent_pos)

        for color, pos in zip(self.ghost_colors, env.ghosts_pos):
            self.draw_pac_object(color, pos)

    def draw_wall(self, color, pos):
        x, y = (pos % self.cell_count, pos // self.cell_count)
        x = x * SIZE
        y = y * SIZE
        pygame.draw.rect(self.surf, color, (x, y, SIZE, SIZE))

    def draw_pac_object(self, color, pos, by=2):
        center = SIZE // 2
        radius = center // by
        x, y = (pos % self.cell_count, pos // self.cell_count)
        x = x * SIZE + center
        y = y * SIZE + center
        pygame.draw.circle(self.surf, color, (x, y), radius)

    def reset(self):
        self.surf.fill((255, 255, 255))
        self.build_rects()


def play_line_world(env: LineWorld, pi_v: PolicyAndValueFunction, method: str = ''):
    WIDTH = SIZE * env.cells_count
    HEIGHT = SIZE
    pygame.init()
    FramePerSec = pygame.time.Clock()

    display_surface = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"LineWorld - {method}")

    gui = LineWorldGui(env.cells_count)

    s = int(env.cells_count / 2)

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        gui.move_agent(s)
        gui.reset()

        display_surface.blit(gui.surf, (0, 0))

        pygame.display.update()
        FramePerSec.tick(60)

        time.sleep(1)

        if env.is_state_terminal(s):
            break

        a = np.argmax(list(pi_v.pi[s].values()))
        print(s, a, pi_v.pi[s])

        if a == 0:
            s -= 1
        else:
            s += 1

    pygame.quit()


def play_grid_world(env: GridWorld, pi_v: PolicyAndValueFunction, method: str = ''):
    print(pi_v)
    WIDTH = SIZE * env.grid_count
    HEIGHT = SIZE * env.grid_count
    pygame.init()
    FramePerSec = pygame.time.Clock()

    display_surface = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"GridWorld - {method}")

    gui = GridWorldGui(env.grid_count)

    s = 0

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        gui.move_agent(s)
        gui.reset()

        display_surface.blit(gui.surf, (0, 0))

        pygame.display.update()
        FramePerSec.tick(60)

        time.sleep(1)

        if env.is_state_terminal(s):
            print('terminal', 24)
            time.sleep(2)
            break

        a = list(pi_v.pi[s].keys())[np.argmax(list(pi_v.pi[s].values()))]
        last_s = s

        if a == 0:
            s -= env.grid_count
        elif a == 1:
            s += env.grid_count
        elif a == 2:
            s -= 1
        else:
            s += 1

        print(last_s, s, a, pi_v.pi[last_s], gui.player_pos)

    pygame.quit()


def play_tictactoe(env: TicTacToe, pi_q: PolicyAndActionValueFunction, method: str = ''):
    print(pi_q)
    WIDTH = SIZE * env.grid_count
    HEIGHT = SIZE * env.grid_count
    pygame.init()
    FramePerSec = pygame.time.Clock()

    display_surface = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"TicTacToe - {method}")

    gui = TicTacToeGui(env.grid_count)

    s = env.state_id()

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        if s not in pi_q.pi:
            a = random.choice(env.available_actions_ids())
        else:
            a = list(pi_q.pi[s].keys())[np.argmax(list(pi_q.pi[s].values()))]

        env.act_with_action_id(a)

        gui.reset()
        gui.draw(env)
        gui.build_rects()

        display_surface.blit(gui.surf, (0, 0))

        pygame.display.update()
        FramePerSec.tick(60)

        time.sleep(1)

        if env.is_game_over():
            print('score', env.score())
            time.sleep(2)
            break

        s = env.state_id()

    pygame.quit()


def play_deep_tictactoe(env: DeepTicTacToe, q: DeepQNetwork, method: str = ''):
    WIDTH = SIZE * env.grid_count
    HEIGHT = SIZE * env.grid_count
    pygame.init()
    FramePerSec = pygame.time.Clock()

    display_surface = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Deep TicTacToe - {method}")

    gui = TicTacToeGui(env.grid_count)

    state_description_length = env.state_description_length()
    max_actions_count = env.max_actions_count()

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        s = env.state_description()
        available_actions = env.available_actions_ids()
        all_q_inputs = np.zeros((len(available_actions), state_description_length + max_actions_count))

        for i, a in enumerate(available_actions):
            all_q_inputs[i] = np.hstack([s, tf.keras.utils.to_categorical(a, max_actions_count)])
        all_q_values = np.squeeze(q.predict(all_q_inputs))

        chosen_action = available_actions[np.argmax(all_q_values)]

        env.act_with_action_id(chosen_action)

        gui.reset()
        gui.draw(env)
        gui.build_rects()

        display_surface.blit(gui.surf, (0, 0))

        pygame.display.update()
        FramePerSec.tick(60)

        time.sleep(1)

        if env.is_game_over():
            print('score', env.score())
            time.sleep(2)
            break

    pygame.quit()


def play_deep_pacman(env: PacMan, q: DeepQNetwork, method: str = ''):
    WIDTH = SIZE * env.grid_count
    HEIGHT = SIZE * env.grid_count
    pygame.init()
    FramePerSec = pygame.time.Clock()

    display_surface = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Deep TicTacToe - {method}")

    gui = PacManGui(env.grid_count)

    state_description_length = env.state_description_length()
    max_actions_count = env.max_actions_count()

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        s = env.state_description()
        available_actions = env.available_actions_ids()
        all_q_inputs = np.zeros((len(available_actions), state_description_length + max_actions_count))

        for i, a in enumerate(available_actions):
            all_q_inputs[i] = np.hstack([s, tf.keras.utils.to_categorical(a, max_actions_count)])
        all_q_values = np.squeeze(q.predict(all_q_inputs))

        # ids = np.where(all_q_values == all_q_values.max())[0]
        # if len(ids) != 1:
        #     chosen_action = available_actions[np.random.choice(ids, 1)[0]]
        # else:
        #     chosen_action = available_actions[np.argmax(all_q_values)]
        chosen_action = available_actions[np.argmax(all_q_values)]
        print(f'Q values: {all_q_values} - Chosen action {chosen_action}')

        env.act_with_action_id(chosen_action)

        gui.reset()
        gui.draw(env)
        gui.build_rects()

        display_surface.blit(gui.surf, (0, 0))

        pygame.display.update()
        FramePerSec.tick(60)

        time.sleep(1)

        if env.is_game_over():
            print('score', env.score())
            time.sleep(2)
            break

    pygame.quit()
