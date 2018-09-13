import numpy as np


class Environment:

    def __init__(self):
        self.board = np.zeros((3, 3))
        self.winner = 0

    def is_empty(self, i, j):
        """
        :param i: Col index
        :param j: Row index
        :return: Bool, True if [i][j] is empty, False otherwise
        """
        return True if self.board[i][j] == 0 else False

    def get_state(self):
        """
        Enumerates through the board to 'see' it, and encodes the current state as a decimal
        :return: Decimal representation of the board's current layout
        """
        flattened_board = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    flattened_board.append(0)
                elif self.board[i][j] == 1:
                    flattened_board.append(1)
                elif self.board[i][j] == -1:
                    flattened_board.append(2)

        state_decimal_form = 0
        for i in range(len(flattened_board)):
            state_decimal_form += (3 ** i) * flattened_board[len(flattened_board) - 1 - i]

    def check_win(self):
        """
        Checks if either player has won
        :return: int 1 if Player 1 wins, int -1 if Player 2 wins, else 0
        """
        for i in range(3):
            if self.board[i][0] + self.board[i][1] + self.board[i][2] == 3:
                self.winner = 1
                return self.winner
            if self.board[i][0] + self.board[i][1] + self.board[i][2] == -3:
                self.winner = -1
                return self.winner
        for j in range(3):
            if self.board[0][j] + self.board[1][j] + self.board[2][j] == 3:
                self.winner = 1
                return self.winner
            if self.board[0][j] + self.board[1][j] + self.board[2][j] == -3:
                self.winner = -1
                return self.winner
        if self.board[0][0] + self.board[1][1] + self.board[2][2] == 3:
            self.winner = 1
            return self.winner
        if self.board[0][0] + self.board[1][1] + self.board[2][2] == -3:
            self.winner = -1
            return self.winner
        if self.board[0][2] + self.board[1][1] + self.board[2][0] == 3:
            self.winner = 1
            return self.winner
        if self.board[0][2] + self.board[1][1] + self.board[2][0] == -3:
            self.winner = -1
            return self.winner
        return False

    def game_over(self):
        """
        Checks if game is over
        :return: 1 or -1 if player 1 or 2 has won (resp.), else False if board not full, else Draw
        """
        self.winner = None

        if self.check_win():
            return self.winner

        for i in range(3):
            for j in range(3):
                if self.is_empty(i,j):
                    return False

        return 'Draw'

    def reward(self, player):
        """
        :param player: Agent we're inspecting
        :return: 1 if agent has won, else 0
        """
        self.game_over()

        if self.winner == player:
            return 1
        else:
            return 0


class Agent:

    def __init__(self, player, learning_rate, epsilon):
        self.player = player
        self.alpha = learning_rate
        self.eps = epsilon
        self.state_history = []
        self.V = None

    def action(self, env):
        """
        Performs a move. If epsilon, goes through all possible placements and chooses a random one.
        If greedy, check value of all empty spaces and for all potential moves:
            Get the state of the potential updated board
            If the value function of this potential state is greater than the current best value:
                Set/update best value as the VF of the potential state
                Set/update next move as current (i,j)

        Updates the board at [next_move[0]][next_move[1]] with self.player
        :param env: Environment class
        :return: Nothing, just updates board
        """
        best_value = -1

        # seek all potential moves
        possible_moves = []
        for i in range(3):
            for j in range(3):
                if env.is_empty(i, j):
                    possible_moves.append((i, j))

        # epsilon
        if np.random.rand() < self.eps:
            next_move = np.random.choice(possible_moves)

        # greedy
        else:
            for potential_move in possible_moves:
                env.board[potential_move[0], potential_move[1]] = self.player
                state = env.get_state()
                env.board[potential_move[0], potential_move[1]] = 0
                if self.V[state] > best_value:
                    best_value = self.V[state]
                    next_move = (potential_move[0], potential_move[1])

        # place counter (self.player) at position (next_move)
        env.board[next_move[0], next_move[1]] = self.player




    def update_history(self, state):
        """
        Adds history to state
        :param state:
        :return:
        """
        self.state_history.append(state)

    def update_value_function(self, env):
        """
        Updates value function after episode
        :param env:
        :return:
        """
        target = env.reward(self.player)
        for last_state in reversed(self.state_history):
            value = self.V[last_state] + self.alpha * (target - self.V[last_state])
            self.V[last_state] = value
            target = value
        self.state_history = []  #resets history


def play_game(p1, p2, env):
    """
    Plays one episode
    :param p1: Player 1
    :param p2: Player 2
    :param env: Game environment
    :return: Nothing - Agents play the game and, after completion, update their value functions
    """

    current_player = p2

    while not env.game_over():
        # switch players
        if current_player == p1:
            current_player = p2
        elif current_player == p2:
            current_player = p1

        # draw board
        pass

        # action
        current_player.action(env)

        # update state history storage
        state = env.get_state()
        p1.update_history(state)
        p2.update_history(state)

    # draw board
    pass

    # value function update
    p1.update_value_function()
    p2.update_value_function()


def get_state_hash_and_winner(env, i=0, j=0):
    """
    We need to create an initial value function. This needs to output 1 when state is win, 0 if draw/lose, 0.5 otherwise.
    To do this, we need to consider EVERY state.
    :return:
    """
    results = []

    for v in (0, env.x, env.o):
        env.board[i, j] = v  # if empty board it should already be 0
        if j == 2:
            # j goes back to 0, increase i, unless i = 2, then we are done
            if i == 2:
                # the board is full, collect results and return
                state = env.get_state()
                ended = env.game_over(force_recalculate=True)
                winner = env.winner
                results.append((state, winner, ended))
            else:
                results += get_state_hash_and_winner(env, i + 1, 0)
        else:
            # increment j, i stays the same
            results += get_state_hash_and_winner(env, i, j + 1)

def initialise(env, state_winner):



#  class Agent:
#     def __init__(self, sym, eps, alpha, V):
#         self.state_history = []
#         self.sym = sym
#         self.eps = eps
#         self.alpha = alpha
#         self.V = V
#
#     def action(self, env):
#
#         #random move
#         if np.random.rand() < self.eps:
#             possible_moves = []
#             for i in range(3):
#                 for j in range(3):
#                     if env.is_spot_empty(i,j):
#                         possible_moves.append((i,j))
#             chosen_spot = np.random.choice(possible_moves)
#
#         #best current move
#         else:
#             for i in range(3):
#                 for j in range(3):
#                     if env.is_spot_empty(i,j):
#                         #consider move
#                         env.board[i,j] = self.sym
#                         state = env.get_state
#
#                         env.board[i,j] = 0
#                         if self.V[state] > best_value:
#                             best_value = self.V[state]
#                             best_state = state
#                             chosen_spot = (i,j)
#
#     def update_state_history(self, state):
#         self.state_history.append(state)
#
#     def update_value_function(self):
#         target = env.reward(self.sym)
#         for last_state in reversed(self.state_history):
#             value = self.V[last_state] + self.alpha * (target - self.V[last_state])
#             self.V[last_state] = value
#             target = value
#         self.state_history = []  #resets history
#
#
# class Environment:
#     """
#
#     """
#     def __init__(self):
#         self.board = np.zeros((3,3))
#         self.winner = None
#
#     def is_spot_empty(self, i, j):
#         if self.board[i][j] == 0:
#             return True
#         else:
#             return False
#
#     def reward(self, sym):
#         if not game_over():
#             return 0
#         elif self.winner == sym:
#             return 1
#         else:
#             return 0.5
#
#     def get_state(self):
#         flattened_board = []
#         for i in range(3):
#             for j in range(3):
#                 if self.board[i][j] == 0:
#                     flattened_board.append(0)
#                 elif self.board[i][j] == 1:
#                     flattened_board.append(1)
#                 elif self.board[i][j] == -1:
#                     flattened_board.append(2)
#
#         state_decimal_form = 0
#         for i in range(flattened_board):
#             state_decimal_form += (3 ** i) * reversed(flattened_board)[i]
#
#     def check_win(self):
#         for i in range(3):
#             if self.board[i][0] + self.board[i][1] + self.board[i][2] == 3:
#                 self.winner = 'o'
#                 return self.winner
#             if self.board[i][0] + self.board[i][1] + self.board[i][2] == -3:
#                 self.winner = 'x'
#                 return self.winner
#         for j in range(3):
#             if self.board[0][j] + self.board[1][j] + self.board[2][j] == 3:
#                 self.winner = 'o'
#                 return self.winner
#             if self.board[0][j] + self.board[1][j] + self.board[2][j] == -3:
#                 self.winner = 'x'
#                 return self.winner
#         if self.board[0][0] + self.board[1][1] + self.board[2][2] == 3:
#             self.winner = 'o'
#             return self.winner
#         if self.board[0][0] + self.board[1][1] + self.board[2][2] == -3:
#             self.winner = 'x'
#             return self.winner
#         return False
#
#     def game_over(self):
#         self.winner = None
#
#         for i in range(3):
#             for j in range(3):
#                 if not self.is_spot_empty(i,j):
#                     return False
#
#         if self.check_win():
#             return self.winner
#
#         return 'Draw'
#
#     def draw_board(self):
#         pass
#
#
# def play_game(p1, p2, env):
#
#     current_player = p2
#
#     while not game_end():
#         #switch players
#         if current_player == p1:
#             current_player = p2
#         elif current_player == p2:
#             current_player = p1
#
#         #draw board
#
#
#         #action
#         current_player.action(env)
#
#         #update state history storage
#         state = env.get_state()
#         p1.update_state(state)
#         p2.update_state(state)
#
#     #draw board
#
#
#     #value function update/backprop
#     p1.update_value_function()
#     p2.update_value_function()

