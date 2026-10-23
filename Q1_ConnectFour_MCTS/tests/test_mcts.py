"""Tests for the MCTS move generator (mcts.py), using tests/test_data/positions.json."""

import json
import math
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from games import ConnectFour, state_from_rows  # noqa: E402
from mcts import (MCT_Node, build_tree, best_move, monte_carlo_tree_search,  # noqa: E402
                  objective, utility, play_game, random_player)

with open(os.path.join(HERE, 'test_data', 'positions.json')) as f:
    DATA = json.load(f)


def all_nodes(root):
    stack = [root]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.children)


class TestObjective(unittest.TestCase):

    def setUp(self):
        self.game = ConnectFour()

    def test_empty_board_is_zero(self):
        self.assertEqual(objective(self.game, self.game.initial, 'X'), 0.0)

    def test_won_and_lost_states(self):
        case = DATA['rules'][0]  # X has won
        s = state_from_rows(self.game, case['rows'])
        self.assertEqual(objective(self.game, s, 'X'), 1)
        self.assertEqual(objective(self.game, s, 'O'), -1)

    def test_draw_is_zero(self):
        s = state_from_rows(self.game, DATA['rules'][4]['rows'])
        self.assertEqual(objective(self.game, s, 'X'), 0.0)

    def test_zero_sum_and_range(self):
        s = state_from_rows(self.game, DATA['rules'][5]['rows'])
        x_val = objective(self.game, s, 'X')
        self.assertAlmostEqual(x_val, -objective(self.game, s, 'O'))
        self.assertTrue(-1 <= x_val <= 1)

    def test_centre_and_threes_score_higher(self):
        centre = self.game.result(self.game.initial, (4, 1))
        edge = self.game.result(self.game.initial, (1, 1))
        self.assertGreater(objective(self.game, centre, 'X'),
                           objective(self.game, edge, 'X'))
        two = state_from_rows(self.game, ['.......'] * 5 + ['XX.....'], 'O')
        three = state_from_rows(self.game, ['.......'] * 5 + ['XXX....'], 'O')
        self.assertGreater(objective(self.game, three, 'X'),
                           objective(self.game, two, 'X'))


class TestUtility(unittest.TestCase):

    def test_unvisited_node_is_infinite(self):
        parent = MCT_Node(N=10)
        self.assertEqual(utility(MCT_Node(parent=parent), C=1.4, W=1.0), math.inf)

    def test_formula(self):
        parent = MCT_Node(N=100)
        n = MCT_Node(parent=parent, U=30, N=50, objective=0.2)
        expected = 30 / 50 + 1.4 * math.sqrt(math.log(100) / 50) + 0.5 * 0.2
        self.assertAlmostEqual(utility(n, C=1.4, W=0.5), expected)

    def test_weights_switch_parts_off(self):
        parent = MCT_Node(N=100)
        n = MCT_Node(parent=parent, U=30, N=50, objective=0.2)
        self.assertAlmostEqual(utility(n, C=0, W=0), 30 / 50)
        self.assertAlmostEqual(utility(n, C=0, W=1) - utility(n, C=0, W=0), 0.2)


class TestMoveGenerator(unittest.TestCase):

    def setUp(self):
        self.game = ConnectFour()

    def test_returns_legal_move(self):
        move = monte_carlo_tree_search(self.game.initial, self.game, cycles=100, seed=0)
        self.assertIn(move, self.game.actions(self.game.initial))

    def test_tactical_positions(self):
        """Finds the winning move and blocks the opponent's winning move."""
        for case in DATA['tactics']:
            s = state_from_rows(self.game, case['rows'], case['to_move'])
            for seed in range(5):
                with self.subTest(case['name'], seed=seed):
                    move = monte_carlo_tree_search(s, self.game, cycles=500, expand_x=7,
                                                   height=3, seed=seed)
                    self.assertEqual(move, tuple(case['expected']))

    def test_number_of_cycles(self):
        for cycles in (1, 10, 250):
            root = build_tree(self.game.initial, self.game, cycles=cycles, seed=0)
            self.assertEqual(root.N, cycles)
            # every cycle passes through exactly one child of the root
            self.assertEqual(sum(c.N for c in root.children), cycles)

    def test_tree_never_deeper_than_height(self):
        for height in (1, 2, 3, 5):
            root = build_tree(self.game.initial, self.game, cycles=300, height=height, seed=0)
            self.assertEqual(max(n.depth for n in all_nodes(root)), height)

    def test_each_cycle_expands_down_to_height(self):
        root = build_tree(self.game.initial, self.game, cycles=1, expand_x=3, height=4, seed=0)
        self.assertEqual(sum(1 for _ in all_nodes(root)), 1 + 3 * 4)

    def test_at_most_x_children_per_node(self):
        for x in (1, 2, 4, 7):
            root = build_tree(self.game.initial, self.game, cycles=300, expand_x=x, seed=0)
            self.assertEqual(len(root.children), x)
            for n in all_nodes(root):
                self.assertLessEqual(len(n.children), x)

    def test_expands_the_x_best_moves_by_objective(self):
        root = build_tree(self.game.initial, self.game, cycles=1, expand_x=3, height=1, seed=0)
        chosen = sorted(c.objective for c in root.children)
        all_values = sorted(objective(self.game, self.game.result(self.game.initial, a), 'X')
                            for a in self.game.actions(self.game.initial))
        self.assertEqual(chosen, all_values[-3:])
        self.assertIn((4, 1), root.children.values())  # centre column is best

    def test_statistics_are_consistent(self):
        root = build_tree(self.game.initial, self.game, cycles=400, seed=1)
        for n in all_nodes(root):
            self.assertTrue(0 <= n.U <= n.N)
            if n.children:
                # a node is visited once more than its children: the cycle that created them
                self.assertLessEqual(sum(c.N for c in n.children), n.N)

    def test_best_move_has_most_playouts(self):
        root = build_tree(self.game.initial, self.game, cycles=300, seed=2)
        move = best_move(root)
        most = max(c.N for c in root.children)
        self.assertTrue(any(c.N == most and a == move for c, a in root.children.items()))

    def test_same_seed_gives_same_move(self):
        s = state_from_rows(self.game, DATA['rules'][5]['rows'])
        moves = {monte_carlo_tree_search(s, self.game, cycles=200, seed=7) for _ in range(3)}
        self.assertEqual(len(moves), 1)

    def test_pure_playout_utility_still_works(self):
        case = DATA['tactics'][0]
        s = state_from_rows(self.game, case['rows'], case['to_move'])
        move = monte_carlo_tree_search(s, self.game, cycles=500, W=0.0, seed=0)
        self.assertEqual(move, tuple(case['expected']))

    def test_beats_random_player(self):
        wins = 0
        for g in range(10):
            rng = random.Random(g)
            mcts = lambda game, state, g=g: monte_carlo_tree_search(state, game, cycles=200, seed=g)
            rand = lambda game, state, rng=rng: random_player(game, state, rng)
            if g % 2 == 0:
                final, me = play_game(self.game, mcts, rand), 'X'
            else:
                final, me = play_game(self.game, rand, mcts), 'O'
            wins += self.game.utility(final, me) == 1
        self.assertGreaterEqual(wins, 9)


if __name__ == '__main__':
    unittest.main()
