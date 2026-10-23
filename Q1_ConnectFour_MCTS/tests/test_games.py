"""Tests for the Connect Four rules (games.py), using tests/test_data/positions.json."""

import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from games import ConnectFour, state_from_rows  # noqa: E402

with open(os.path.join(HERE, 'test_data', 'positions.json')) as f:
    DATA = json.load(f)


class TestConnectFourRules(unittest.TestCase):

    def setUp(self):
        self.game = ConnectFour()

    def test_board_is_6_rows_by_7_columns(self):
        self.assertEqual(self.game.h, 7)  # columns
        self.assertEqual(self.game.v, 6)  # rows
        self.assertEqual(self.game.k, 4)
        self.assertEqual(len(self.game.initial.moves), 42)

    def test_initial_state(self):
        s = self.game.initial
        self.assertEqual(self.game.to_move(s), 'X')
        self.assertEqual(sorted(self.game.actions(s)), [(x, 1) for x in range(1, 8)])
        self.assertFalse(self.game.terminal_test(s))

    def test_piece_drops_to_lowest_empty_row(self):
        s = self.game.result(self.game.initial, (4, 1))
        self.assertEqual(s.board, {(4, 1): 'X'})
        self.assertEqual(self.game.to_move(s), 'O')
        self.assertIn((4, 2), self.game.actions(s))
        self.assertNotIn((4, 1), self.game.actions(s))
        self.assertNotIn((4, 3), self.game.actions(s))

    def test_players_alternate(self):
        s = self.game.initial
        for move, player in [((1, 1), 'X'), ((1, 2), 'O'), ((2, 1), 'X')]:
            self.assertEqual(self.game.to_move(s), player)
            s = self.game.result(s, move)
            self.assertEqual(s.board[move], player)

    def test_illegal_move_has_no_effect(self):
        s = self.game.result(self.game.initial, (4, 1))
        self.assertIs(self.game.result(s, (4, 1)), s)

    def test_positions_from_test_data(self):
        for case in DATA['rules']:
            with self.subTest(case['name']):
                s = state_from_rows(self.game, case['rows'], case.get('to_move'))
                self.assertEqual(self.game.terminal_test(s), case['terminal'])
                self.assertEqual(self.game.utility(s, 'X'), case['utility_X'])
                self.assertEqual(self.game.utility(s, 'O'), -case['utility_X'])
                if 'legal_moves' in case:
                    expected = sorted(tuple(m) for m in case['legal_moves'])
                    self.assertEqual(sorted(self.game.actions(s)), expected)

    def test_winning_move_ends_game(self):
        moves = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (4, 1)]
        s = self.game.initial
        for m in moves:
            self.assertFalse(self.game.terminal_test(s))
            s = self.game.result(s, m)
        self.assertTrue(self.game.terminal_test(s))
        self.assertEqual(self.game.utility(s, 'X'), 1)
        self.assertEqual(self.game.actions(s), [])


if __name__ == '__main__':
    unittest.main()
