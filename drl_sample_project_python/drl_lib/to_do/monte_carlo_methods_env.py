import random

from ..do_not_touch.contracts import SingleAgentEnv
import numpy as np


class AdversaryPlayerEnum:
    HUMAN = 'human'
    BOT = 'bot'


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
        self.agent_player_id = 0
        self.adv_player_id = 0
        self.player_ids = [1, 2]
        self.reset()
        self.__setup_board_to_state()
        self.__setup_action_id_to_coord()
        self.__select_first_player()

    def __select_first_player(self):
        """
        Define who plays first. Id=1 is the first to play.
        """
        if np.random.randn(1) > 0.5:
            self.agent_player_id, self.adv_player_id = self.player_ids
        else:
            self.adv_player_id, self.agent_player_id = self.player_ids

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

        ids = self.available_actions_ids()
        if len(ids) != 0:
            self.current_step += 1
            self.__play(self.adv_player_id)

        if self.current_step >= 5:
            if self.__check_win(self.agent_player_id):
                self.game_over = True
                self.current_score = 1.0

            if self.__check_win(self.adv_player_id):
                self.game_over = True
                self.current_score = -1.0

        if (not self.game_over) and ((self.current_step >= self.max_steps) or len(self.available_actions_ids()) == 0):
            self.current_score = -1.0
            self.game_over = True

    def __play(self, player_id, action_id=None):
        assert player_id in self.player_ids

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
        self.__select_first_player()

    def reset_random(self):
        self.map = np.zeros(self.max_cells)
        self.agent_pos = np.random.randint(0, self.grid_count)
        self.game_over = False
        self.current_score = 0.0
        self.current_step = 0
        self.__select_first_player()
