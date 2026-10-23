"""Monte Carlo tree search move generator for Connect Four.

Based on MONTE-CARLO-TREE-SEARCH, textbook Figure 5.11:

    function MONTE-CARLO-TREE-SEARCH(state) returns an action
        tree <- NODE(state)
        while IS-TIME-REMAINING() do
            leaf   <- SELECT(tree)
            child  <- EXPAND(leaf)
            result <- SIMULATE(child)
            BACK-PROPAGATE(result, child)
        return the move in ACTIONS(state) whose node has highest number of playouts

and on the MCT_Node / ucb / monte_carlo_tree_search code that accompanies
the text (aima-python games4e.py). Changes required by the assignment:

  * SELECT uses a utility with a playout part (U/N plus the UCB1
    exploration term, from the stored results of earlier playouts) and an
    objective part (an evaluation function computed from the rules of
    the game, as in Section 5.4).
  * EXPAND adds x children at each level of the tree (the x most
    promising moves by the objective part).
  * The tree has a fixed height. Each cycle keeps expanding down to that
    level, then SIMULATE plays a random playout from there.
  * The loop repeats a configurable number of cycles instead of
    IS-TIME-REMAINING().
"""

import math
import random

from games import opponent

# Points for a window of four cells that holds only one player's pieces,
# keyed by how many of that player's pieces it holds (Section 5.4: a
# weighted linear function of features).
WINDOW_WEIGHTS = {1: 1, 2: 3, 3: 9}
CENTER_WEIGHT = 1
OBJECTIVE_SCALE = 50.0


class MCT_Node:
    """Node in the Monte Carlo search tree, keeps track of the children states.
    U = total utility of the playouts through this node and N = number of
    playouts, both from the point of view of the player who made the move
    into this node."""

    def __init__(self, parent=None, state=None, U=0, N=0, depth=0, objective=0.0):
        self.parent = parent
        self.state = state
        self.U = U
        self.N = N
        self.depth = depth
        self.objective = objective
        self.children = {}  # child node -> action that leads to it


def windows(game):
    """All lines of k cells on the board (horizontal, vertical, both diagonals)."""
    lines = []
    for x in range(1, game.h + 1):
        for y in range(1, game.v + 1):
            for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
                cells = [(x + i * dx, y + i * dy) for i in range(game.k)]
                if all(1 <= cx <= game.h and 1 <= cy <= game.v for cx, cy in cells):
                    lines.append(cells)
    return lines


def objective(game, state, player, lines=None):
    """Objective part of the utility: an evaluation function built from the
    rules of the game, from player's point of view, in the range [-1, 1].

    A won state is +1, a lost state -1 and a drawn state 0. Otherwise it is
    a weighted linear function of features (Section 5.4): for every line of
    four that only one player can still complete, that player gets points
    for the pieces already in it (1 for one, 3 for two, 9 for three), and
    each piece in the centre column is worth 1. The opponent's points are subtracted."""
    if state.utility != 0:
        return game.utility(state, player)
    if not state.moves:
        return 0.0
    if lines is None:
        lines = windows(game)
    board = state.board
    other = opponent(player)
    score = 0
    for cells in lines:
        pieces = [board.get(c) for c in cells]
        mine = pieces.count(player)
        theirs = pieces.count(other)
        if theirs == 0:
            score += WINDOW_WEIGHTS.get(mine, 0)
        elif mine == 0:
            score -= WINDOW_WEIGHTS.get(theirs, 0)
    center = (game.h + 1) // 2
    for y in range(1, game.v + 1):
        piece = board.get((center, y))
        if piece == player:
            score += CENTER_WEIGHT
        elif piece == other:
            score -= CENTER_WEIGHT
    return max(-1.0, min(1.0, score / OBJECTIVE_SCALE))


def utility(n, C=1.4, W=1.0):
    """Selection utility of node n.
    Playout part: U/N + C * sqrt(ln N(parent) / N)  (UCB1, Section 5.5).
    Objective part: W * objective(n).
    An unvisited node gets infinity, as in UCB1, so it is tried first."""
    if n.N == 0:
        return math.inf
    playout = n.U / n.N + C * math.sqrt(math.log(n.parent.N) / n.N)
    return playout + W * n.objective


def build_tree(state, game, cycles=1000, expand_x=7, height=3, C=1.4, W=1.0, seed=None):
    """Run the MCTS cycles from state and return the root of the tree.

    cycles   -- number of SELECT / EXPAND / SIMULATE / BACK-PROPAGATE cycles
    expand_x -- number of children added when a node is expanded (x)
    height   -- fixed level of the tree at which the random playout starts
    C        -- exploration constant of the playout part
    W        -- weight of the objective part
    seed     -- seed for the random number generator (None = not fixed)
    """
    rng = random.Random(seed)
    lines = windows(game)

    def best_child(n):
        # Highest utility; ties (e.g. several unvisited children) are broken
        # by the objective part, then at random.
        return max(n.children, key=lambda c: (utility(c, C, W), c.objective, rng.random()))

    def select(n):
        while n.children:
            n = best_child(n)
        return n

    def expand(n):
        while n.depth < height and not game.terminal_test(n.state):
            if not n.children:
                mover = game.to_move(n.state)
                candidates = []
                for action in game.actions(n.state):
                    child_state = game.result(n.state, action)
                    candidates.append((objective(game, child_state, mover, lines),
                                       rng.random(), action, child_state))
                candidates.sort(reverse=True)
                for obj, _, action, child_state in candidates[:expand_x]:
                    child = MCT_Node(parent=n, state=child_state,
                                     depth=n.depth + 1, objective=obj)
                    n.children[child] = action
            n = best_child(n)
        return n

    def simulate(state):
        # Random playout policy; returns the utility for 'X' (+1, 0 or -1).
        while not game.terminal_test(state):
            action = rng.choice(game.actions(state))
            state = game.result(state, action)
        return game.utility(state, 'X')

    def backprop(n, result):
        while n is not None:
            mover = opponent(game.to_move(n.state))  # player who moved into n
            value = result if mover == 'X' else -result
            n.U += (value + 1) / 2  # win = 1, draw = 0.5, loss = 0
            n.N += 1
            n = n.parent

    root = MCT_Node(state=state)
    for _ in range(cycles):
        leaf = select(root)
        child = expand(leaf)
        result = simulate(child.state)
        backprop(child, result)
    return root


def best_move(root):
    """The move whose node has the highest number of playouts (Figure 5.11)."""
    child = max(root.children, key=lambda c: (c.N, c.U))
    return root.children[child]


def monte_carlo_tree_search(state, game, cycles=1000, expand_x=7, height=3,
                            C=1.4, W=1.0, seed=None):
    """Return the move chosen by MCTS for the player to move in state."""
    root = build_tree(state, game, cycles, expand_x, height, C, W, seed)
    return best_move(root)


def mcts_player(game, state, **config):
    return monte_carlo_tree_search(state, game, **config)


def random_player(game, state, rng=random):
    """A player that chooses a legal move at random."""
    return rng.choice(game.actions(state))


def play_game(game, x_player, o_player, state=None):
    """Play a game; x_player and o_player are functions (game, state) -> move.
    Returns the final state."""
    state = game.initial if state is None else state
    while not game.terminal_test(state):
        player = x_player if game.to_move(state) == 'X' else o_player
        state = game.result(state, player(game, state))
    return state


if __name__ == '__main__':
    import argparse
    from games import ConnectFour, state_from_rows

    parser = argparse.ArgumentParser(description='MCTS move generator for Connect Four.')
    parser.add_argument('--cycles', type=int, default=1000, help='number of MCTS cycles')
    parser.add_argument('--x', type=int, default=7, help='nodes expanded at each level')
    parser.add_argument('--height', type=int, default=3, help='fixed height of the tree')
    parser.add_argument('--C', type=float, default=1.4, help='exploration constant')
    parser.add_argument('--W', type=float, default=1.0, help='weight of the objective part')
    parser.add_argument('--seed', type=int, default=None, help='random seed (default: not fixed)')
    parser.add_argument('--rows', nargs=6, metavar='ROW',
                        help='board as 6 strings, top row first, e.g. ....... ... XXO....')
    args = parser.parse_args()

    game = ConnectFour()
    state = state_from_rows(game, args.rows) if args.rows else game.initial
    game.display(state)
    root = build_tree(state, game, args.cycles, args.x, args.height, args.C, args.W, args.seed)
    print('\nmove    N      U/N   objective')
    for child, action in sorted(root.children.items(), key=lambda item: item[1]):
        print(f'{str(action):7} {child.N:<6} {child.U / child.N:.3f}  {child.objective:+.3f}')
    print(f'\n{game.to_move(state)} plays {best_move(root)}')
