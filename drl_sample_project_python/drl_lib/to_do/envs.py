import os

import numpy as np

from .helper import WEIGHT_PATH
from ..do_not_touch.contracts import *


class AdversaryPlayerEnum:
    HUMAN = 'human'
    BOT = 'bot'


class LineWorld(MDPEnv):
    def __init__(self, cells_count: int):
        assert (cells_count >= 3)
        self.cells_count = cells_count
        self.S = np.arange(self.cells_count)
        self.A = np.array([0, 1])
        self.R = np.array([-1, 0, 1])
        self.p = np.zeros((len(self.S), len(self.A), len(self.S), len(self.R)))
        self.build_env_dynamic()

    def build_env_dynamic(self):
        for i in range(1, self.cells_count - 2):
            self.p[i, 1, i + 1, 1] = 1.0

        for i in range(2, self.cells_count - 1):
            self.p[i, 0, i - 1, 1] = 1.0

        self.p[self.cells_count - 2, 1, self.cells_count - 1, 2] = 1.0
        self.p[1, 0, 0, 0] = 1.0

    def states(self) -> np.ndarray:
        return self.S

    def actions(self) -> np.ndarray:
        return self.A

    def rewards(self) -> np.ndarray:
        return self.R

    def is_state_terminal(self, s: int) -> bool:
        return s in [0, self.cells_count-1]

    def transition_probability(self, s: int, a: int, s_p: int, r: float) -> float:
        return self.p[s, a, s_p, r]

    def view_state(self, s: int):
        print("State {} is : {}".format(s, self.p[s]))


class GridWorld(MDPEnv):
    def __init__(self, grid_count: int):
        assert (grid_count >= 3)
        self.grid_count = grid_count
        self.max_cells = self.grid_count*self.grid_count
        self.S = np.arange(self.max_cells)
        self.A = np.array([0, 1, 2, 3])  # 0:up, 1: down, 2: left, 3: right
        self.R = np.array([-1, 0, 1])
        self.p = np.zeros((len(self.S), len(self.A), len(self.S), len(self.R)))
        self.build_env_dynamic()

    def build_env_dynamic(self):
        max_cells = self.max_cells
        # up
        for i in range(self.grid_count, max_cells - 1):
            self.p[i, 0, i - self.grid_count, 1] = 1.0
        # down
        for i in range(max_cells - self.grid_count):
            if i == self.grid_count - 1:
                continue
            self.p[i, 1, i + self.grid_count, 1] = 1.0
        # left
        for i in range(1, max_cells - 1):
            if i % self.grid_count == 0 or i == self.grid_count - 1:
                continue
            self.p[i, 2, i-1, 1] = 1.0
        # right
        for i in range(max_cells - 1):
            if (i+1) % self.grid_count == 0:
                continue
            self.p[i, 3, i+1, 1] = 1.0

        # Go to upper rightmost from neighbours
        self.p[self.grid_count - 2, 3, self.grid_count - 1, 0] = 1.0
        self.p[2*self.grid_count - 1, 0, self.grid_count - 1, 0] = 1.0
        # Go to lower rightmost from neighbours
        self.p[max_cells - self.grid_count - 1, 1, max_cells - 1, 2] = 1.0
        self.p[max_cells - 2, 3, max_cells - 1, 2] = 1.0

    def states(self) -> np.ndarray:
        return self.S

    def actions(self) -> np.ndarray:
        return self.A

    def rewards(self) -> np.ndarray:
        return self.R

    def is_state_terminal(self, s: int) -> bool:
        return s in [self.grid_count - 1, self.max_cells - 1]

    def transition_probability(self, s: int, a: int, s_p: int, r: float) -> float:
        return self.p[s, a, s_p, r]

    def view_state(self, s: int):
        print("State {} is : {}".format(s, self.p[s]))


class TicTacToe(SingleAgentEnv):
    def __init__(self, grid_count: int = 3, adv=AdversaryPlayerEnum.BOT):
        assert (grid_count >= 3)
        self.grid_count = grid_count
        self.max_cells = self.grid_count ** 2
        self.max_steps = self.max_cells
        self.map = np.zeros(self.max_cells)
        self.board_to_state = {}
        self.action_id_to_coord = {}
        self.agent_pos = 0
        self.game_over = False
        self.adv = adv
        self.current_score = 0.0
        self.current_step = 0
        self.agent_player_id = 1
        self.adv_player_id = 2
        self.reset()
        self.__setup_board_to_state()
        self.__setup_action_id_to_coord()
        # self.__select_first_player()

    def __setup_board_to_state(self):
        self.board_to_state = {}
        for i in range(19683):
            s = i
            board = ''
            for j in range(9):
                board += str(s % 3)
                s //= 3
            self.board_to_state[board] = i

    def __map_to_board(self):
        board = ''
        for i in self.map:
            board += str(int(i))
        return board

    def __setup_action_id_to_coord(self):
        self.action_id_to_coord = {}
        for line in range(self.grid_count):
            for col in range(self.grid_count):
                action_id = line * self.grid_count + col
                self.action_id_to_coord[action_id] = (line, col)

    def state_id(self) -> int:
        return self.board_to_state[self.__map_to_board()]

    def is_game_over(self) -> bool:
        return self.game_over

    def act_with_action_id(self, action_id: int):
        assert (0 <= action_id <= self.max_cells) and self.map[action_id] == 0, action_id
        assert (not self.game_over)

        self.current_step += 1
        self.__play(self.agent_player_id, action_id)

        if self.current_step >= 5:
            if self.__check_win(self.agent_player_id):
                self.game_over = True
                self.current_score = 1.0

        ids = self.available_actions_ids()
        if len(ids) != 0 and not self.is_game_over():
            self.current_step += 1
            self.__play(self.adv_player_id)

            if self.current_step >= 5:
                if self.__check_win(self.adv_player_id):
                    self.game_over = True
                    self.current_score = -1.0

        if (not self.game_over) and ((self.current_step >= self.max_steps) or len(self.available_actions_ids()) == 0):
            self.current_score = 0.0
            self.game_over = True

    def __play(self, player_id, action_id=None):
        assert player_id in [1, 2]

        if player_id == self.agent_player_id:
            assert action_id is not None
            self.map[action_id] = self.agent_player_id

        if player_id == self.adv_player_id:
            action = np.random.choice(self.available_actions_ids())
            if self.adv == AdversaryPlayerEnum.HUMAN:
                self.view()
                action = int(input('What is your move: '))

            self.map[action] = self.adv_player_id

    def __check_win(self, player_id):
        game_map = np.copy(self.map)
        player_game_pos = np.where(game_map == player_id)[0]

        has_won = False
        for i, a1 in enumerate(player_game_pos[:-1]):
            for j, a2 in enumerate(player_game_pos):
                if (i == j) or (j == i + 1) or has_won:
                    continue

                p1 = self.action_id_to_coord[a1]
                p2 = self.action_id_to_coord[a2]
                p3 = self.action_id_to_coord[player_game_pos[i + 1]]

                # Vectoriel product
                p1p2 = (p2[0] - p1[0], p2[1] - p1[1])
                p2p3 = (p2[0] - p3[0], p2[1] - p3[1])
                vp = p1p2[0] * p2p3[1] - p1p2[1] * p2p3[0]

                if vp == 0:
                    has_won = True
                    break

        return has_won

    def score(self) -> float:
        return self.current_score

    def available_actions_ids(self) -> np.ndarray:
        if self.game_over:
            return np.array([], dtype=np.int)
        return np.where(self.map == 0)[0]

    def view(self):
        board = self.__map_to_board()
        print()
        for i in range(self.grid_count):
            print(board[i * self.grid_count:(i + 1) * self.grid_count])
        print()

    def reset(self):
        center = int(np.ceil(self.grid_count / 2))
        self.map = np.zeros(self.max_cells)
        self.game_over = False
        self.current_score = 0.0
        self.agent_pos = center
        self.current_step = 0

    def reset_random(self):
        self.map = np.zeros(self.max_cells)
        self.agent_pos = np.random.randint(0, self.grid_count)
        self.game_over = False
        self.current_score = 0.0
        self.current_step = 0


class DeepTicTacToe(DeepSingleAgentWithDiscreteActionsEnv):
    def __init__(self, grid_count: int = 3, adv=AdversaryPlayerEnum.BOT):
        assert (grid_count >= 3)
        self.grid_count = grid_count
        self.max_cells = self.grid_count ** 2
        self.max_steps = self.max_cells
        self.map = np.zeros(self.max_cells)
        self.board_to_state = {}
        self.action_id_to_coord = {}
        self.game_over = False
        self.adv = adv
        self.current_score = 0.0
        self.current_step = 0
        self.agent_player_id = 1
        self.adv_player_id = 2
        self.reset()
        self.__setup_board_to_state()
        self.__setup_action_id_to_coord()

    def __setup_board_to_state(self):
        self.board_to_state = {}
        for i in range(19683):
            s = i
            board = ''
            for j in range(9):
                board += str(s % 3)
                s //= 3
            self.board_to_state[board] = i

    def __map_to_board(self):
        board = ''
        for i in self.map:
            board += str(int(i))
        return board

    def __setup_action_id_to_coord(self):
        self.action_id_to_coord = {}
        for line in range(self.grid_count):
            for col in range(self.grid_count):
                action_id = line * self.grid_count + col
                self.action_id_to_coord[action_id] = (line, col)

    def state_description(self) -> np.ndarray:
        return self.map

    def state_description_length(self) -> int:
        return self.max_cells

    def max_actions_count(self) -> int:
        return self.max_cells

    def is_game_over(self) -> bool:
        return self.game_over

    def act_with_action_id(self, action_id: int):
        assert (0 <= action_id <= self.max_cells) and self.map[action_id] == 0, action_id
        assert (not self.game_over)

        self.current_step += 1
        self.__play(self.agent_player_id, action_id)

        if self.current_step >= 5:
            if self.__check_win(self.agent_player_id):
                self.game_over = True
                self.current_score = 1.0

        ids = self.available_actions_ids()
        if len(ids) != 0 and not self.is_game_over():
            self.current_step += 1
            self.__play(self.adv_player_id)

            if self.current_step >= 5:
                if self.__check_win(self.adv_player_id):
                    self.game_over = True
                    self.current_score = -1.0

        if (not self.game_over) and ((self.current_step >= self.max_steps) or len(self.available_actions_ids()) == 0):
            self.current_score = 0.0
            self.game_over = True

    def __play(self, player_id, action_id=None):
        assert player_id in [1, 2]

        if player_id == self.agent_player_id:
            assert action_id is not None
            self.map[action_id] = self.agent_player_id

        if player_id == self.adv_player_id:
            action = np.random.choice(self.available_actions_ids())
            if self.adv == AdversaryPlayerEnum.HUMAN:
                self.view()
                action = int(input('What is your move: '))

            self.map[action] = self.adv_player_id

    def __check_win(self, player_id):
        game_map = np.copy(self.map)
        player_game_pos = np.where(game_map == player_id)[0]

        has_won = False
        for i, a1 in enumerate(player_game_pos[:-1]):
            for j, a2 in enumerate(player_game_pos):
                if (i == j) or (j == i + 1) or has_won:
                    continue

                p1 = self.action_id_to_coord[a1]
                p2 = self.action_id_to_coord[a2]
                p3 = self.action_id_to_coord[player_game_pos[i + 1]]

                # Vectoriel product
                p1p2 = (p2[0] - p1[0], p2[1] - p1[1])
                p2p3 = (p2[0] - p3[0], p2[1] - p3[1])
                vp = p1p2[0] * p2p3[1] - p1p2[1] * p2p3[0]

                if vp == 0:
                    has_won = True
                    break

        return has_won

    def score(self) -> float:
        return self.current_score

    def available_actions_ids(self) -> np.ndarray:
        if self.game_over:
            return np.array([], dtype=np.int)
        return np.where(self.map == 0)[0]

    def view(self):
        board = self.__map_to_board()
        print()
        for i in range(self.grid_count):
            print(board[i * self.grid_count:(i + 1) * self.grid_count])
        print()

    def reset(self):
        self.map = np.zeros(self.max_cells)
        self.game_over = False
        self.current_score = 0.0
        self.current_step = 0

    def reset_random(self):
        self.map = np.zeros(self.max_cells)
        self.game_over = False
        self.current_score = 0.0
        self.current_step = 0


class PacMan(DeepSingleAgentWithDiscreteActionsEnv):
    def __init__(self, grid_count: int = 20, wall_pct: float = 0.15):
        assert (grid_count >= 10) and (wall_pct <= 0.2)
        self.grid_count = grid_count
        self.wall_pct = wall_pct
        self.max_cells = self.grid_count ** 2
        self.map = None
        self.reshaped_map = None
        self.game_over = False
        self.current_score = 0.0
        self.agent_pos = self.max_cells-1  # bottom right
        self.ghosts_pos = []  # center of the map
        self.nb_ghosts = 4
        self.entity_ids = {"void": 0, "food": 1, "wall": 2, "ghost": 3, "pacman": 4}
        self.action_ids = [0, 1, 2, 3]  # up, down, left, right
        self.reset()  # rebuild the env

    def __set_ghosts_start_pos(self):
        center = (self.max_cells // 2) - (self.grid_count // 2)
        self.ghosts_pos = [center - 1, center, center + self.grid_count - 1, center + self.grid_count]

    def build_map(self):
        map_name = f'{WEIGHT_PATH}pacman_env_size_{self.grid_count}_{self.wall_pct}.npy'
        if os.path.exists(map_name):
            self.map = np.load(map_name)
            # To avoid BC break
            for pos in self.ghosts_pos:
                self.map[pos] = self.entity_ids['void']

            self.reshaped_map = self.map.reshape((self.grid_count, self.grid_count))
            return

        self.map = np.ones(self.max_cells)

        # Pacman
        self.map[self.agent_pos] = self.entity_ids['pacman']

        # Ghosts
        for pos in self.ghosts_pos:
            self.map[pos] = self.entity_ids['ghost']

        # Walls
        wall_counts = int(self.wall_pct * self.max_cells)
        for w in range(wall_counts):
            ids = np.where(self.map == 1)[0]
            # take out border to avoid trapped rewards
            ids = [idx for idx in ids if (idx > self.grid_count) and (idx < self.max_cells - self.grid_count) and (idx % self.grid_count != 0)]
            ids = [idx for idx in ids if (idx+1) % self.grid_count != 0]
            pos = np.random.choice(ids, 1)
            self.map[pos] = self.entity_ids['wall']

        # Remove Ghosts from map
        # This avoid erasing food position when the ghosts move to a new positions
        for pos in self.ghosts_pos:
            self.map[pos] = self.entity_ids['void']

        self.reshaped_map = self.map.reshape((self.grid_count, self.grid_count))

        np.save(map_name, self.map)

    def state_description(self) -> np.ndarray:
        map_with_ghosts = np.copy(self.map)
        for pos in self.ghosts_pos:
            map_with_ghosts[pos] = self.entity_ids['ghost']
        return map_with_ghosts

    def state_description_length(self) -> int:
        return self.max_cells

    def max_actions_count(self) -> int:
        return len(self.action_ids)

    def is_game_over(self) -> bool:
        return self.game_over

    def action_to_pos(self, action, pos):
        if action == 0:
            return pos - self.grid_count
        if action == 1:
            return pos + self.grid_count
        if action == 2:
            return pos - 1
        if action == 3:
            return pos + 1

    def get_potential_moves(self, pos):
        possible_actions = self.action_ids
        if pos < self.grid_count:
            possible_actions = [a for a in possible_actions if a != 0]  # up is invalid
        if pos > self.max_cells - self.grid_count:
            possible_actions = [a for a in possible_actions if a != 1]  # down is invalid
        if pos == 0 or (pos % self.grid_count == 0):
            possible_actions = [a for a in possible_actions if a != 2]  # left is invalid
        if (pos+1) % self.grid_count == 0:
            possible_actions = [a for a in possible_actions if a != 3]  # right is invalid

        ids = [(a_id, self.action_to_pos(a_id, pos)) for a_id in possible_actions]
        possible_moves = {a_id: pos for a_id, pos in ids if (0 <= pos < self.max_cells) and self.map[pos] != self.entity_ids['wall']}

        return possible_moves

    def act_with_action_id(self, action_id: int):
        new_pos = self.action_to_pos(action_id, self.agent_pos)

        assert (0 <= new_pos < self.max_cells) and self.map[new_pos] != self.entity_ids['wall'], f'Action {action_id} is invalid. Result in pos {new_pos}'
        assert (not self.game_over)

        if self.map[new_pos] == self.entity_ids['food']:
            self.current_score += 1.0

        self.map[self.agent_pos] = self.entity_ids['void']
        self.map[new_pos] = self.entity_ids['pacman']
        self.agent_pos = new_pos

        new_ghosts_pos = []
        for pos in self.ghosts_pos:
            moves = self.get_potential_moves(pos)

            new_ghosts_pos.append(np.random.choice(list(moves.values()), 1)[0])
        self.ghosts_pos = new_ghosts_pos

        food_map = np.where(self.map == self.entity_ids['food'])[0]
        if len(food_map) == 0:
            self.game_over = True

        if not self.game_over:
            pacman_surrounding = self.get_potential_moves(self.agent_pos).values()
            dead = len(set(pacman_surrounding).intersection(set(self.ghosts_pos))) != 0

            if dead:
                self.game_over = True

    def score(self) -> float:
        return self.current_score

    def available_actions_ids(self) -> np.ndarray:
        if self.game_over:
            return np.array([], dtype=np.int)
        return np.array(list(self.get_potential_moves(self.agent_pos).keys()))

    def view(self):
        w, h = self.reshaped_map.shape

        for j in range(h):
            print('', end='|')
            for i in range(w):
                val = self.reshaped_map[j, i]
                str_ = ' {:^3} '

                if val == self.entity_ids["food"]:
                    str_ = str_.format('.')
                elif val == self.entity_ids["wall"]:
                    str_ = str_.format('#')
                elif val == self.entity_ids["ghost"]:
                    str_ = str_.format('x')
                elif val == self.entity_ids["pacman"]:
                    str_ = str_.format('c')
                else:
                    str_ = str_.format(' ')

                print(str_, end='|')
            print()

    def reset(self):
        self.game_over = False
        self.current_score = 0.0
        self.agent_pos = self.max_cells-1
        self.__set_ghosts_start_pos()
        self.build_map()
