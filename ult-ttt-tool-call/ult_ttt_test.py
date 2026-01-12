from unittest import TestCase
import numpy as np

from ult_ttt import *

class TestState(TestCase):
    def test_actions(self):
        state = ImmutableState(board=np.zeros((3, 3, 3, 3)), fill_num=1, prev_local_action=None)
        actions = get_all_valid_actions(state)
        self.assertEqual(set(actions),
                         set((i, j, k, l) for i in range(3) for j in range(3) for k in range(3) for l in range(3)))
        for action in actions:
            self.assertTrue(is_valid_action(state, action))
        prev_action = (1, 1, 1, 1)
        state = change_state(state, prev_action)
        actions = get_all_valid_actions(state)
        self.assertEqual(set(actions),
                         set((1, 1, i, j) for i in range(3) for j in range(3) if (i, j) != (1, 1)))
        for action in actions:
            self.assertTrue(is_valid_action(state, action))
        prev_action = (1, 1, 0, 0)
        state = change_state(state, prev_action)
        actions = get_all_valid_actions(state)
        self.assertEqual(set(actions),
                         set((0, 0, i, j) for i in range(3) for j in range(3)))
        for action in actions:
            self.assertTrue(is_valid_action(state, action))
        prev_action = (0, 0, 1, 1)
        state = change_state(state, prev_action)
        actions = get_all_valid_actions(state)
        self.assertEqual(set(actions),
                         set((1, 1, i, j) for i in range(3) for j in range(3) if (i, j) != (1, 1) and (i, j) != (0, 0)))
        for action in actions:
            self.assertTrue(is_valid_action(state, action))

        state = ImmutableState(board=np.array([
            [
                [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
                [[0, 2, 2], [0, 1, 1], [0, 2, 2]],
                [[0, 2, 0], [1, 2, 1], [2, 1, 2]],
            ],
            [
                [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
                [[2, 2, 2], [0, 0, 0], [0, 0, 0]],
                [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
            ],
            [
                [[2, 2, 2], [0, 0, 0], [0, 0, 0]],
                [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
                [[2, 2, 0], [0, 0, 0], [2, 2, 2]],
            ],
        ]), fill_num=1, prev_local_action=(2, 2))
        actions = get_all_valid_actions(state)
        self.assertEqual(set(actions), {(0, 1, 0, 0), (0, 1, 1, 0), (0, 1, 2, 0), (0, 2, 0, 0), (0, 2, 0, 2)})
        for action in actions:
            self.assertTrue(is_valid_action(state, action))
    
    def test_function(self):
        state_str = """
X  X  X  | -  O  O  | -  O  -
-  -  -  | -  X  X  | X  O  X
-  -  -  | -  O  O  | O  X  O
------------------------------
X  X  X  | O  O  O  | X  X  X
-  -  -  | -  -  -  | -  -  -
-  -  -  | -  -  -  | -  -  -
------------------------------
O  O  O  | X  X  X  | O  O  -
-  -  -  | -  -  -  | -  -  -
-  -  -  | -  -  -  | O  O  O
"""
        prev_move = 81
        self.assertEqual(set(get_valid_moves(state_str, prev_move)), {10, 13, 16, 19, 21})

    def test_local_board_status(self):
        state = ImmutableState(board=np.array([
            [
                [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
                [[0, 2, 2], [0, 1, 1], [0, 2, 2]],
                [[0, 2, 0], [1, 2, 1], [2, 1, 2]],
            ],
            [
                [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
                [[2, 2, 2], [0, 0, 0], [0, 0, 0]],
                [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
            ],
            [
                [[2, 2, 2], [0, 0, 0], [0, 0, 0]],
                [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
                [[2, 2, 0], [0, 0, 0], [2, 2, 2]],
            ],
        ]), prev_local_action=None, fill_num=1)
        self.assertTrue(np.all(state.local_board_status == np.array([[1, 0, 0], [1, 2, 1], [2, 1, 2]])))

        state = ImmutableState(board=np.array([
            [
                [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
                [[0, 2, 2], [0, 1, 1], [0, 2, 2]],
                [[1, 2, 1], [1, 2, 1], [2, 1, 2]],
            ],
            [
                [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
                [[2, 2, 2], [0, 0, 0], [0, 0, 0]],
                [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
            ],
            [
                [[2, 2, 2], [0, 0, 0], [0, 0, 0]],
                [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
                [[2, 2, 0], [0, 0, 0], [2, 2, 2]],
            ],
        ]), prev_local_action=None, fill_num=1)
        self.assertTrue(np.all(state.local_board_status == np.array([[1, 0, 3], [1, 2, 1], [2, 1, 2]])))

    def test_winner(self):
        state = ImmutableState(board=np.array([
            [
                [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
                [[0, 2, 2], [0, 1, 1], [0, 2, 2]],
                [[0, 2, 0], [1, 2, 1], [2, 1, 2]],
            ],
            [
                [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
                [[2, 2, 2], [0, 0, 0], [0, 0, 0]],
                [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
            ],
            [
                [[2, 2, 2], [0, 0, 0], [0, 0, 0]],
                [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
                [[2, 2, 0], [0, 0, 0], [2, 2, 2]],
            ],
        ]), fill_num=2, prev_local_action=None)
        self.assertFalse(is_terminal(state))
        state = change_state(state, (0, 2, 0, 2))
        self.assertTrue(is_terminal(state))
        self.assertEqual(board_status(state.local_board_status), 2)

        state = ImmutableState(board=np.array([
            [
                [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
                [[1, 2, 2], [2, 1, 1], [1, 2, 2]],
                [[1, 2, 1], [1, 2, 1], [2, 1, 2]],
            ],
            [
                [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
                [[2, 2, 2], [0, 0, 0], [0, 0, 0]],
                [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
            ],
            [
                [[2, 2, 2], [0, 0, 0], [0, 0, 0]],
                [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
                [[2, 2, 0], [0, 0, 0], [2, 2, 2]],
            ],
        ]), fill_num=2, prev_local_action=None)
        self.assertTrue(is_terminal(state))
        self.assertEqual(board_status(state.local_board_status), 3)

    def test_clone(self):
        state = ImmutableState(board=np.array([
            [
                [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
                [[0, 2, 2], [0, 1, 1], [0, 2, 2]],
                [[0, 2, 0], [1, 2, 1], [2, 1, 2]],
            ],
            [
                [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
                [[2, 2, 2], [0, 0, 0], [0, 0, 0]],
                [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
            ],
            [
                [[2, 2, 2], [0, 0, 0], [0, 0, 0]],
                [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
                [[2, 2, 0], [0, 0, 0], [2, 2, 2]],
            ],
        ]), fill_num=2, prev_local_action=None)
        state_clone = clone_state(state)
        self.assertTrue(np.all(state.board == state_clone.board))
        self.assertTrue(np.all(state.local_board_status == state_clone.local_board_status))
        self.assertEqual(state.fill_num, state_clone.fill_num)
        self.assertEqual(state, state_clone)
    
    def test_board_to_string(self):
        board = np.array([
            [
                [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
                [[0, 2, 2], [0, 1, 1], [0, 2, 2]],
                [[0, 2, 0], [1, 2, 1], [2, 1, 2]],
            ],
            [
                [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
                [[2, 2, 2], [0, 0, 0], [0, 0, 0]],
                [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
            ],
            [
                [[2, 2, 2], [0, 0, 0], [0, 0, 0]],
                [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
                [[2, 2, 0], [0, 0, 0], [2, 2, 2]],
            ],
        ])
        expected_output = (
            "X  X  X  | -  O  O  | -  O  -\n"
            "-  -  -  | -  X  X  | X  O  X\n"
            "-  -  -  | -  O  O  | O  X  O\n"
            "------------------------------\n"
            "X  X  X  | O  O  O  | X  X  X\n"
            "-  -  -  | -  -  -  | -  -  -\n"
            "-  -  -  | -  -  -  | -  -  -\n"
            "------------------------------\n"
            "O  O  O  | X  X  X  | O  O  -\n"
            "-  -  -  | -  -  -  | -  -  -\n"
            "-  -  -  | -  -  -  | O  O  O"
        )
        self.assertEqual(convert_board_to_string(board), expected_output)
    
    def test_string_to_board(self):
        string = (
            "X  X  X  | -  O  O  | -  O  -\n"
            "-  -  -  | -  X  X  | X  O  X\n"
            "-  -  -  | -  O  O  | O  X  O\n"
            "------------------------------\n"
            "X  X  X  | O  O  O  | X  X  X\n"
            "-  -  -  | -  -  -  | -  -  -\n"
            "-  -  -  | -  -  -  | -  -  -\n"
            "------------------------------\n"
            "O  O  O  | X  X  X  | O  O  -\n"
            "-  -  -  | -  -  -  | -  -  -\n"
            "-  -  -  | -  -  -  | O  O  O"
        )
        expected_board = board = np.array([
            [
                [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
                [[0, 2, 2], [0, 1, 1], [0, 2, 2]],
                [[0, 2, 0], [1, 2, 1], [2, 1, 2]],
            ],
            [
                [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
                [[2, 2, 2], [0, 0, 0], [0, 0, 0]],
                [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
            ],
            [
                [[2, 2, 2], [0, 0, 0], [0, 0, 0]],
                [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
                [[2, 2, 0], [0, 0, 0], [2, 2, 2]],
            ],
        ])
        self.assertTrue(np.all(convert_string_to_board(string) == expected_board))
