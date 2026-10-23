from collections import namedtuple

GameState = namedtuple('GameState', 'to_move, utility, board, moves')


class Game:
    def actions(self, state):
        raise NotImplementedError

    def result(self, state, move):
        raise NotImplementedError

    def utility(self, state, player):
        raise NotImplementedError

    def terminal_test(self, state):
        return not self.actions(state)

    def to_move(self, state):
        return state.to_move

    def display(self, state):
        print(state)


class TicTacToe(Game):
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
            return state
        board = state.board.copy()
        board[move] = state.to_move
        moves = list(state.moves)
        moves.remove(move)
        return GameState(to_move=('O' if state.to_move == 'X' else 'X'),
                         utility=self.compute_utility(board, move, state.to_move),
                         board=board, moves=moves)

    def utility(self, state, player):
        return state.utility if player == 'X' else -state.utility

    def terminal_test(self, state):
        return state.utility != 0 or len(state.moves) == 0

    def display(self, state):
        board = state.board
        for y in range(self.v, 0, -1):
            print(' '.join(board.get((x, y), '.') for x in range(1, self.h + 1)))
        print(' '.join(str(x) for x in range(1, self.h + 1)))

    def compute_utility(self, board, move, player):
        if (self.k_in_row(board, move, player, (0, 1)) or
                self.k_in_row(board, move, player, (1, 0)) or
                self.k_in_row(board, move, player, (1, -1)) or
                self.k_in_row(board, move, player, (1, 1))):
            return +1 if player == 'X' else -1
        return 0

    def k_in_row(self, board, move, player, delta_x_y):
        (delta_x, delta_y) = delta_x_y
        x, y = move
        n = 0
        while board.get((x, y)) == player:
            n += 1
            x, y = x + delta_x, y + delta_y
        x, y = move
        while board.get((x, y)) == player:
            n += 1
            x, y = x - delta_x, y - delta_y
        n -= 1
        return n >= self.k


class ConnectFour(TicTacToe):
    def __init__(self, h=7, v=6, k=4):
        TicTacToe.__init__(self, h, v, k)

    def actions(self, state):
        if state.utility != 0:
            return []
        return [(x, y) for (x, y) in state.moves
                if y == 1 or (x, y - 1) in state.board]


def opponent(player):
    return 'O' if player == 'X' else 'X'


def state_from_rows(game, rows, to_move=None):
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
