import datetime
import os

import numpy
import torch
from .abstract_game import AbstractGame
from random import choice


EMPTY = 0

class WormZeroConfig:
    def __init__(self):
        self.seed = 0
        self.max_num_gpus = None

        ### Game
        self.observation_shape = (3, 6, 7)
        self.action_space = list(range(7))
        self.players = list(range(2))
        self.stacked_observations = 2

        # Evaluate
        self.wormzero_player = 0
        self.opponent = "negamax"

        # Self-Play
        self.num_workers = 16
        self.selfplay_on_gpu = False
        self.max_moves = 42
        self.num_simulations = 50
        self.temperature_threshold = None   # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time


        # Root prior exploration_noise
        self.root_dirichlet_alpha = 1
        self.root_exploration_fraction = 0.75

        # Network
        self.network = "resnet"
        #TODO


        # Training
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results", os.path.basename(__file__)[:-3], datetime.datetime.now.strftime("%Y-%m-%d--%H-%M-%S"))
        self.save_model = True
        self.training_steps = 500000
        self.batch_size = 256
        self.checkpoint_interval = 100
        self.train_on_gpu = torch.cuda.is_available()

        self.optimizer = "Adamw"
        self.weight_decay = 1e-4
        self.momentum = 0.9

        # Expoential learning rate schedule
        self.lr_init = 0.02
        self.lr_decay_rate = 0.3
        self.lr_decay_steps = 50000
        

        # Replay Buffer
        self.replay_buffer_size = 200000
        
        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False

        # Adjust the self play/ training ratio to avoid over/underfitting
        self.self_play_delay = 0
        self.training_delay = 0
        self.ratio = 2    # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it


    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        return 1

class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = Connect4()

    def step(self, action):
        """
        Apply action to the game.
        
        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config. 
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.
        
        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.
        
        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        choice = input(f"Enter the column to play for the player {self.to_play()}: ")
        while choice not in [str(action) for action in self.legal_actions()]:
            choice = input("Enter another column : ")
        return int(choice)

    def expert_agent(self):
        """
        Hard coded agent that WormZero faces to assess his progress in multiplayer games.
        It doesn't influence training
        Now it plays second by default, it means that it maybe tend to prevent WormZero from winning if it plays first
        Returns:
            Action as an integer to take in the current game state
        """
        return self.env.expert_action()

    def negamax_agent(self, max_depth=4):
        """
        adapted form kaggle_environment
        """
        _, column = self.negamax(self.env.board, self.env.player, max_depth)
        return column

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        return f"Play column {action_number + 1}"


class Connect4:
    def __init__(self):
        self.board = numpy.zeros(42, dtype="int32")
        self.steps = 1
        self.player = 1

    def to_play(self):
        return 0 if self.player == 1 else 1
    
    def play(self, board, column, player):
        row = max([r for r in range(6) if board[column + (r * 7)] == EMPTY])
        board[column + (row * 7)] = player

    def reset(self):
        self.board = numpy.zeros(42, dtype="int32")
        self.steps = 1
        self.player = 1
        return self.get_observation()

    def step(self, action):
        self.steps += 1
        # variable reward
        reward = 1.18 - (9*self.steps/350) if self.is_win(self.board, action, self.player, False) else 0
        
        done = self.is_win(self.board, action, self.player, False) or len(self.legal_actions()) == 0
        self.play(self.board, action, self.player)
        self.player *= -1

        return self.get_observation(), reward, done

    def get_observation(self):
        board_player1 = numpy.where(self.board == 1, 1.0, 0.0).reshape((6,7))
        board_player2 = numpy.where(self.board == -1, 1.0, 0.0).reshape((6,7))
        board_to_play = numpy.full((6, 7), self.player, dtype="int32")
        return numpy.array([board_player1, board_player2, board_to_play])

    def legal_actions(self):
        legal = [c for c in range(7) if self.board[c] == EMPTY]
        return legal

    def is_win(self, board, column, player, has_played=True):
        row = (
            min([r for r in range(6) if board[column + (r * 7)] == player])
            if has_played
            else max([r for r in range(6) if board[column + (r * 7)] == EMPTY])
        )
        
        def count(offset_row, offset_column):
            for i in range(1, 4):
                r = row + offset_row * i
                c = column + offset_column * i
                if (
                    r < 0
                    or r >= 6
                    or c < 0
                    or c >= 7
                    or board[c + (r*7)] != player
                ):
                    return i - 1
            return 3
        
        return (
            count(1, 0) >= 3
            or (count(0, 1) + count(0, -1)) >= 3
            or (count(-1, -1) + count(1, 1)) >= 3
            or (count(-1, 1) + count(1, -1)) >= 3 
        )

    def expert_action(self):
        board = self.board
        legal_actions = self.legal_actions()
        action = numpy.random.choice(legal_actions)
        for legal_action in legal_actions:
            if self.is_win(board, legal_action, self.player, False) or self.is_win(board, legal_action, -self.player, False):
                return legal_action

        return action

    def negamax(self, board, player, depth):
        moves = sum(1 if cell != EMPTY else 0 for cell in board)
        
        # Tie
        if moves == 42:
            return (0, None)
        
        # can win next
        for column in range(7):
            if board[column] == EMPTY and self.is_win(board, column, player, False):
                return ((43 - moves) / 2, column)
        
        best_score = -42
        best_column = None
        for column in range(7):
            if board[column] == EMPTY:
                if depth <= 0:
                    row = max(
                        [
                            r
                            for r in range(6)
                            if board[column + (r * 7)] == EMPTY
                        ]
                    )
                    score = (43 - moves) / 2
                    if column > 0 and board[row * 7 + column - 1] == player:
                        score += 1
                    if (
                        column < 6
                        and board[row * 7 + column + 1] == player
                    ):
                        score += 1
                    if row > 0 and board[(row - 1) * 7 + column] == player:
                        score += 1
                    if row < 4 and board[(row + 1) * 7 + column] == player:
                        score += 1
                else:
                    next_board = board[:]
                    self.play(next_board, column, player)
                    (score, _) = self.negamax(next_board, -player, depth - 1)
                    score = score * -1
                if score > best_score or (score == best_score and choice([True, False])):
                    best_score = score
                    best_column = column
        
        if best_column == None:
            column = choice([c for c in range(7) if self.board[c] == EMPTY])
        return (best_score, best_column)

    def render(self):
        matrix_board = self.board.reshape((6, 7))
        print(matrix_board[::-1])
