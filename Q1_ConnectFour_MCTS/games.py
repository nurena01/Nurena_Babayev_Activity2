"""Connect Four game definition.

Follows the Game / TicTacToe / ConnectFour classes from the textbook's code
(aima-python games.py, Chapter 5): a game is defined by its initial state,
TO-MOVE, ACTIONS, RESULT, TERMINAL-TEST and UTILITY.

Board coordinates are (x, y): x = column 1..7 (left to right),
y = row 1..6 (bottom to top). A piece dropped in a column lands on the
lowest empty row.
"""

from collections import namedtuple

GameState = namedtuple('GameState', 'to_move, utility, board, moves')


class Game:
    """A game is similar to a problem, but it has a terminal test instead of
    a goal test, and a utility for each terminal state (textbook Section 5.1)."""

    def actions(self, state):
        """Return a list of the allowable moves at this point."""
        raise NotImplementedError

    def result(self, state, move):
        """Return the state that results from making a move from a state."""
        raise NotImplementedError

    def utility(self, state, player):
        """Return the value of this final state to player."""
        raise NotImplementedError

    def terminal_test(self, state):
        """Return True if this is a final state for the game."""
        return not self.actions(state)

    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.to_move

    def display(self, state):
        print(state)


class TicTacToe(Game):
    """k-in-a-row on an h x v board. X moves first. A state's utility is
    +1 if X has won, -1 if O has won, 0 otherwise."""

    def __init__(self, h=3, v=3, k=3):
        self.h = h
        self.v = v
        self.k = k
        moves = [(x, y) for x in range(1, h + 1) for y in range(1, v + 1)]
        self.initial = GameState(to_move='X', utility=0, board={}, moves=moves)

    def actions(self, state):
        return state.moves

    def result(self, state, move):
        if move not in state.moves:
            return state  # Illegal move has no effect
        board = state.board.copy()
        board[move] = state.to_move
        moves = list(state.moves)
        moves.remove(move)
        return GameState(to_move=('O' if state.to_move == 'X' else 'X'),
                         utility=self.compute_utility(board, move, state.to_move),
                         board=board, moves=moves)

    def utility(self, state, player):
        """Return the value to player; 1 for win, -1 for loss, 0 otherwise."""
        return state.utility if player == 'X' else -state.utility

    def terminal_test(self, state):
        """A state is terminal if it is won or there are no empty squares."""
        return state.utility != 0 or len(state.moves) == 0

    def display(self, state):
        board = state.board
        for y in range(self.v, 0, -1):
            print(' '.join(board.get((x, y), '.') for x in range(1, self.h + 1)))
        print(' '.join(str(x) for x in range(1, self.h + 1)))

    def compute_utility(self, board, move, player):
        """If 'X' wins with this move, return 1; if 'O' wins return -1; else return 0."""
        if (self.k_in_row(board, move, player, (0, 1)) or
                self.k_in_row(board, move, player, (1, 0)) or
                self.k_in_row(board, move, player, (1, -1)) or
                self.k_in_row(board, move, player, (1, 1))):
            return +1 if player == 'X' else -1
        return 0

    def k_in_row(self, board, move, player, delta_x_y):
        """Return true if there is a line through move on board for player."""
        (delta_x, delta_y) = delta_x_y
        x, y = move
        n = 0  # n is number of moves in row
        while board.get((x, y)) == player:
            n += 1
            x, y = x + delta_x, y + delta_y
        x, y = move
        while board.get((x, y)) == player:
            n += 1
            x, y = x - delta_x, y - delta_y
        n -= 1  # Because we counted move itself twice
        return n >= self.k


class ConnectFour(TicTacToe):
    """A TicTacToe-like game in which you can only make a move on the bottom
    row, or in a square directly above an occupied square. Traditionally
    played on a 7 column x 6 row board, needing 4 in a row."""

    def __init__(self, h=7, v=6, k=4):
        TicTacToe.__init__(self, h, v, k)

    def actions(self, state):
        if state.utility != 0:
            return []  # The game is already won
        return [(x, y) for (x, y) in state.moves
                if y == 1 or (x, y - 1) in state.board]


def opponent(player):
    return 'O' if player == 'X' else 'X'


def state_from_rows(game, rows, to_move=None):
    """Build a GameState from a list of v strings, top row first.
    Each character is 'X', 'O' or '.', and column 1 is the leftmost
    character. Used to load test data and experiment positions.
    If to_move is None it is worked out from the piece counts (X moves first)."""
    board = {}
    for i, row in enumerate(rows):
        y = game.v - i
        for j, ch in enumerate(row):
            if ch in 'XO':
                board[(j + 1, y)] = ch
    if to_move is None:
        xs = sum(1 for p in board.values() if p == 'X')
        os_ = sum(1 for p in board.values() if p == 'O')
        to_move = 'X' if xs == os_ else 'O'
    utility = 0
    for move, player in board.items():
        u = game.compute_utility(board, move, player)
        if u != 0:
            utility = u
            break
    moves = [(x, y) for x in range(1, game.h + 1) for y in range(1, game.v + 1)
             if (x, y) not in board]
    return GameState(to_move=to_move, utility=utility, board=board, moves=moves)
