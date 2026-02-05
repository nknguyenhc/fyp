from unittest import TestCase

from xiangqi import Xiangqi, Rook, Horse, Elephant, Advisor, King, Cannon, Pawn, Move

class TestXiangqi(TestCase):

    def test_king_move(self):
        board = Xiangqi(board=[
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Pawn(False), None, None, None, None],
            [None, None, None, None, King(True), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (8, 4), (9, 4)),
            Move(King, (8, 4), (7, 4)),
            Move(King, (8, 4), (8, 5)),
        })

        board = Xiangqi(board=[
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, King(True), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
        ], turn=False)
        self.assertEqual(set(board.actions()), {
            Move(King, (0, 3), (1, 3)),
        })

        board = Xiangqi(board=[
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, King(True), None, None, None],
            [None, None, None, None, None, None, None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (8, 5), (9, 5)),
            Move(King, (8, 5), (7, 5)),
            Move(King, (8, 5), (8, 4)),
        })

        board = Xiangqi(board=[
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, King(True), None, None, None],
            [None, None, None, None, None, None, None, None, None],
        ], turn=False)
        self.assertEqual(set(board.actions()), {
            Move(King, (0, 3), (1, 3)),
            Move(King, (0, 3), (0, 4)),
        })

        board = Xiangqi(board=[
            [None, None, None, None, King(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Pawn(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, King(True), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (8, 3), (9, 3)),
            Move(King, (8, 3), (7, 3)),
            Move(King, (8, 3), (8, 4)),
        })

        board = Xiangqi(board=[
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Pawn(True), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, King(True), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
        ], turn=False)
        self.assertEqual(set(board.actions()), {
            Move(King, (1, 3), (0, 3)),
            Move(King, (1, 3), (2, 3)),
            Move(King, (1, 3), (1, 4)),
        })

    def test_advisor_move(self):
        board = Xiangqi(board=[
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, Advisor(True), King(True), Advisor(True), None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(Advisor, (9, 5), (8, 4)),
            Move(Advisor, (9, 3), (8, 4)),
            Move(King, (9, 4), (8, 4)),
        })

        board = Xiangqi(board=[
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Advisor(True), None, None, None, None],
            [None, None, None, None, King(True), Advisor(True), None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(Advisor, (8, 4), (7, 3)),
            Move(Advisor, (8, 4), (7, 5)),
            Move(Advisor, (8, 4), (9, 3)),
        })

        board = Xiangqi(board=[
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, Advisor(False), None, None, None, None],
            [None, None, None, None, None, Pawn(True), None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, King(True), None, None, None, None],
        ], turn=False)
        self.assertEqual(set(board.actions()), {
            Move(Advisor, (1, 4), (2, 3)),
            Move(Advisor, (1, 4), (2, 5)),
            Move(Advisor, (1, 4), (0, 5)),
            Move(King, (0, 3), (0, 4)),
            Move(King, (0, 3), (1, 3)),
        })

    def test_elephant_move(self):
        board = Xiangqi(board=[
            [None, None, None, None, King(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, King(True), Elephant(True), None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (9, 5), (8, 5)),
            Move(Elephant, (9, 6), (7, 8)),
            Move(Elephant, (9, 6), (7, 4)),
        })

        board = Xiangqi(board=[
            [None, None, None, None, King(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, King(True), None, None, None],
            [None, None, None, None, None, None, Elephant(True), None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (8, 5), (9, 5)),
            Move(King, (8, 5), (7, 5)),
            Move(Elephant, (9, 6), (7, 8)),
        })

        board = Xiangqi(board=[
            [None, None, None, None, King(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, Pawn(False), None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, Elephant(True)],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, King(True), None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (9, 5), (8, 5)),
            Move(Elephant, (7, 8), (9, 6)),
            Move(Elephant, (7, 8), (5, 6)),
        })

        board = Xiangqi(board=[
            [None, None, None, None, King(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Elephant(True), None, None, None, None],
            [None, None, None, None, None, King(True), None, None, None],
            [None, None, None, None, None, None, None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (8, 5), (7, 5)),
            Move(King, (8, 5), (9, 5)),
            Move(King, (8, 5), (8, 4)),
            Move(Elephant, (7, 4), (9, 2)),
            Move(Elephant, (7, 4), (5, 2)),
            Move(Elephant, (7, 4), (5, 6)),
        })

        board = Xiangqi(board=[
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, Elephant(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, King(True), None, None, None, None],
        ], turn=False)
        self.assertEqual(set(board.actions()), {
            Move(King, (1, 3), (2, 3)),
            Move(King, (1, 3), (0, 3)),
            Move(King, (1, 3), (1, 4)),
            Move(Elephant, (2, 4), (0, 6)),
            Move(Elephant, (2, 4), (4, 2)),
            Move(Elephant, (2, 4), (4, 6)),
        })

    def test_horse_move(self):
        board = Xiangqi(board=[
            [None, None, None, None, King(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, Horse(True), None, None, None, None, None],
            [None, Pawn(False), None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, King(True), None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (9, 5), (8, 5)),
            Move(Horse, (5, 3), (7, 4)),
            Move(Horse, (5, 3), (6, 5)),
            Move(Horse, (5, 3), (4, 5)),
            Move(Horse, (5, 3), (3, 4)),
            Move(Horse, (5, 3), (3, 2)),
            Move(Horse, (5, 3), (4, 1)),
            Move(Horse, (5, 3), (6, 1)),
            Move(Horse, (5, 3), (7, 2)),
        })

        board = Xiangqi(board=[
            [None, None, None, None, King(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, Pawn(False), Horse(True), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, King(True), None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (9, 5), (8, 5)),
            Move(Horse, (5, 3), (7, 4)),
            Move(Horse, (5, 3), (6, 5)),
            Move(Horse, (5, 3), (4, 5)),
            Move(Horse, (5, 3), (3, 4)),
            Move(Horse, (5, 3), (3, 2)),
            Move(Horse, (5, 3), (7, 2)),
        })

        board = Xiangqi(board=[
            [None, None, None, None, King(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, King(True), Horse(True), None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (9, 5), (8, 5)),
            Move(Horse, (9, 6), (7, 5)),
            Move(Horse, (9, 6), (7, 7)),
            Move(Horse, (9, 6), (8, 8)),
        })

        board = Xiangqi(board=[
            [None, None, None, None, King(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, Horse(True), None, None],
            [None, None, None, None, None, King(True), None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (9, 5), (8, 5)),
            Move(Horse, (8, 6), (9, 4)),
            Move(Horse, (8, 6), (7, 4)),
            Move(Horse, (8, 6), (6, 5)),
            Move(Horse, (8, 6), (6, 7)),
            Move(Horse, (8, 6), (7, 8)),
            Move(Horse, (8, 6), (9, 8)),
        })

    def test_rook_move(self):
        board = Xiangqi(board=[
            [None, None, None, None, King(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, Rook(True), None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, King(True), None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (9, 5), (8, 5)),
            Move(Rook, (7, 6), (8, 6)),
            Move(Rook, (7, 6), (9, 6)),
            Move(Rook, (7, 6), (6, 6)),
            Move(Rook, (7, 6), (5, 6)),
            Move(Rook, (7, 6), (4, 6)),
            Move(Rook, (7, 6), (3, 6)),
            Move(Rook, (7, 6), (2, 6)),
            Move(Rook, (7, 6), (1, 6)),
            Move(Rook, (7, 6), (0, 6)),
            Move(Rook, (7, 6), (7, 7)),
            Move(Rook, (7, 6), (7, 8)),
            Move(Rook, (7, 6), (7, 5)),
            Move(Rook, (7, 6), (7, 4)),
            Move(Rook, (7, 6), (7, 3)),
            Move(Rook, (7, 6), (7, 2)),
            Move(Rook, (7, 6), (7, 1)),
            Move(Rook, (7, 6), (7, 0)),
        })

        board = Xiangqi(board=[
            [None, None, None, None, King(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, Pawn(False), None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, King(True), None, None, Rook(True), None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (7, 3), (8, 3)),
            Move(Rook, (7, 6), (8, 6)),
            Move(Rook, (7, 6), (9, 6)),
            Move(Rook, (7, 6), (6, 6)),
            Move(Rook, (7, 6), (5, 6)),
            Move(Rook, (7, 6), (4, 6)),
            Move(Rook, (7, 6), (3, 6)),
            Move(Rook, (7, 6), (7, 7)),
            Move(Rook, (7, 6), (7, 8)),
            Move(Rook, (7, 6), (7, 5)),
            Move(Rook, (7, 6), (7, 4)),
        })

    def test_cannon_move(self):
        board = Xiangqi(board=[
            [None, None, None, None, King(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, Cannon(True), None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, King(True), None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (9, 5), (8, 5)),
            Move(Cannon, (7, 6), (8, 6)),
            Move(Cannon, (7, 6), (9, 6)),
            Move(Cannon, (7, 6), (6, 6)),
            Move(Cannon, (7, 6), (5, 6)),
            Move(Cannon, (7, 6), (4, 6)),
            Move(Cannon, (7, 6), (3, 6)),
            Move(Cannon, (7, 6), (2, 6)),
            Move(Cannon, (7, 6), (1, 6)),
            Move(Cannon, (7, 6), (0, 6)),
            Move(Cannon, (7, 6), (7, 7)),
            Move(Cannon, (7, 6), (7, 8)),
            Move(Cannon, (7, 6), (7, 5)),
            Move(Cannon, (7, 6), (7, 4)),
            Move(Cannon, (7, 6), (7, 3)),
            Move(Cannon, (7, 6), (7, 2)),
            Move(Cannon, (7, 6), (7, 1)),
            Move(Cannon, (7, 6), (7, 0)),
        })

        board = Xiangqi(board=[
            [None, None, None, None, King(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, Pawn(False), None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [Pawn(False), None, None, None, None, None, Cannon(True), None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, King(True), None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (9, 5), (8, 5)),
            Move(Cannon, (7, 6), (8, 6)),
            Move(Cannon, (7, 6), (9, 6)),
            Move(Cannon, (7, 6), (6, 6)),
            Move(Cannon, (7, 6), (5, 6)),
            Move(Cannon, (7, 6), (4, 6)),
            Move(Cannon, (7, 6), (7, 7)),
            Move(Cannon, (7, 6), (7, 8)),
            Move(Cannon, (7, 6), (7, 5)),
            Move(Cannon, (7, 6), (7, 4)),
            Move(Cannon, (7, 6), (7, 3)),
            Move(Cannon, (7, 6), (7, 2)),
            Move(Cannon, (7, 6), (7, 1)),
        })

        board = Xiangqi(board=[
            [None, None, None, None, King(False), None, Elephant(False), None, None],
            [None, None, None, None, None, None, Cannon(False), None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, Pawn(False), None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [Pawn(False), None, None, None, None, King(True), Cannon(True), None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (7, 5), (8, 5)),
            Move(Cannon, (7, 6), (8, 6)),
            Move(Cannon, (7, 6), (9, 6)),
            Move(Cannon, (7, 6), (6, 6)),
            Move(Cannon, (7, 6), (5, 6)),
            Move(Cannon, (7, 6), (4, 6)),
            Move(Cannon, (7, 6), (1, 6)),
            Move(Cannon, (7, 6), (7, 7)),
            Move(Cannon, (7, 6), (7, 8)),
            Move(Cannon, (7, 6), (7, 0)),
        })

        board = Xiangqi(board=[
            [None, None, None, None, King(False), None, Elephant(False), None, None],
            [None, None, None, None, None, None, Pawn(True), None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, Pawn(False), None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [Pawn(False), None, None, None, None, King(True), Cannon(True), None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (7, 5), (8, 5)),
            Move(Cannon, (7, 6), (8, 6)),
            Move(Cannon, (7, 6), (9, 6)),
            Move(Cannon, (7, 6), (6, 6)),
            Move(Cannon, (7, 6), (5, 6)),
            Move(Cannon, (7, 6), (4, 6)),
            Move(Cannon, (7, 6), (7, 7)),
            Move(Cannon, (7, 6), (7, 8)),
            Move(Cannon, (7, 6), (7, 0)),
            Move(Pawn, (1, 6), (1, 5)),
            Move(Pawn, (1, 6), (1, 7)),
            Move(Pawn, (1, 6), (0, 6)),
        })

    def test_pawn_move(self):
        board = Xiangqi(board=[
            [None, None, None, None, King(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, Pawn(True), None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [Pawn(True), None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, King(True), None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (9, 5), (8, 5)),
            Move(Pawn, (3, 6), (3, 7)),
            Move(Pawn, (3, 6), (3, 5)),
            Move(Pawn, (3, 6), (2, 6)),
            Move(Pawn, (6, 0), (5, 0)),
        })

        board = Xiangqi(board=[
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, Pawn(False), None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [Pawn(False), None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, King(True), None, None, None, None],
        ], turn=False)
        self.assertEqual(set(board.actions()), {
            Move(King, (0, 3), (1, 3)),
            Move(Pawn, (3, 6), (4, 6)),
            Move(Pawn, (6, 0), (6, 1)),
            Move(Pawn, (6, 0), (7, 0)),
        })

    def test_rook_discover_check_constraint(self):
        board = Xiangqi(board=[
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Rook(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Advisor(True), None, None, None, None],
            [None, None, None, None, King(True), None, None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (9, 4), (9, 5)),
        })

        board = Xiangqi(board=[
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Rook(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Cannon(True), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, King(True), None, None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (9, 4), (9, 5)),
            Move(King, (9, 4), (8, 4)),
            Move(Cannon, (7, 4), (8, 4)),
            Move(Cannon, (7, 4), (6, 4)),
            Move(Cannon, (7, 4), (5, 4)),
            Move(Cannon, (7, 4), (4, 4)),
        })

        board = Xiangqi(board=[
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, King(True), None, Rook(True), Rook(False), None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (9, 4), (8, 4)),
            Move(King, (9, 4), (9, 5)),
            Move(Rook, (9, 6), (9, 5)),
            Move(Rook, (9, 6), (9, 7)),
        })

        board = Xiangqi(board=[
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, King(True), Horse(False), None, Rook(False), None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (9, 4), (8, 4)),
        })

        board = Xiangqi(board=[
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Rook(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Elephant(True), None, None, None, None],
            [None, None, None, None, Advisor(True), None, None, None, None],
            [None, None, None, None, King(True), None, None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (9, 4), (9, 5)),
            Move(Advisor, (8, 4), (9, 3)),
            Move(Advisor, (8, 4), (9, 5)),
            Move(Advisor, (8, 4), (7, 3)),
            Move(Advisor, (8, 4), (7, 5)),
            Move(Elephant, (7, 4), (5, 2)),
            Move(Elephant, (7, 4), (5, 6)),
            Move(Elephant, (7, 4), (9, 2)),
            Move(Elephant, (7, 4), (9, 6)),
        })

    def test_cannon_discover_two_pieces_check_constraint(self):
        board = Xiangqi(board=[
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Cannon(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Elephant(True), None, None, None, None],
            [None, None, None, None, Advisor(True), None, None, None, None],
            [None, None, None, None, King(True), None, None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (9, 4), (9, 5)),
        })

        board = Xiangqi(board=[
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Cannon(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Elephant(True), None, None, None, None],
            [None, None, None, None, Advisor(True), None, None, None, None],
            [None, None, None, Horse(True), King(True), Advisor(True), Elephant(True), Cannon(False), None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(Horse, (9, 3), (7, 2)),
            Move(Horse, (9, 3), (8, 1)),
        })

        board = Xiangqi(board=[
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Cannon(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Rook(True), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Advisor(True), None, None, None, None],
            [None, None, None, None, King(True), Advisor(True), None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(Rook, (6, 4), (7, 4)),
            Move(Rook, (6, 4), (5, 4)),
            Move(Rook, (6, 4), (4, 4)),
        })
        
        board = Xiangqi(board=[
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Cannon(False), None, None, None, None],
            [None, None, None, None, Rook(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Rook(True), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, King(True), None, None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (9, 4), (9, 5)),
            Move(King, (9, 4), (8, 4)),
            Move(Rook, (7, 4), (8, 4)),
            Move(Rook, (7, 4), (6, 4)),
        })

        board = Xiangqi(board=[
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Cannon(False), None, None, None, None],
            [None, None, None, None, Elephant(True), None, None, None, None],
            [None, None, None, None, Horse(False), None, None, None, None],
            [None, None, None, None, King(True), Advisor(True), None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(Advisor, (9, 5), (8, 4))
        })

    def test_cannon_discover_no_piece_check_constraint(self):
        board = Xiangqi(board=[
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, Pawn(True), None, None, None, None, None],
            [None, None, None, None, Cannon(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, Elephant(True), Advisor(True), King(True), Advisor(True), None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (9, 4), (8, 4)),
            Move(Pawn, (3, 3), (3, 2)),
            Move(Pawn, (3, 3), (3, 4)),
            Move(Pawn, (3, 3), (2, 3)),
            Move(Elephant, (9, 2), (7, 0)),
        })

        board = Xiangqi(board=[
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, Rook(True), None, None, Elephant(True)],
            [None, None, None, None, Advisor(True), None, None, None, None],
            [None, None, None, Advisor(True), King(True), None, Cannon(False), None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (9, 4), (9, 5)),
            Move(Advisor, (8, 4), (7, 3)),
            Move(Rook, (7, 5), (8, 5)),
            Move(Rook, (7, 5), (6, 5)),
            Move(Rook, (7, 5), (5, 5)),
            Move(Rook, (7, 5), (4, 5)),
            Move(Rook, (7, 5), (3, 5)),
            Move(Rook, (7, 5), (2, 5)),
            Move(Rook, (7, 5), (1, 5)),
            Move(Rook, (7, 5), (0, 5)),
            Move(Rook, (7, 5), (7, 4)),
            Move(Rook, (7, 5), (7, 3)),
            Move(Rook, (7, 5), (7, 2)),
            Move(Rook, (7, 5), (7, 1)),
            Move(Rook, (7, 5), (7, 0)),
            Move(Rook, (7, 5), (7, 6)),
            Move(Rook, (7, 5), (7, 7)),
            Move(Elephant, (7, 8), (5, 6)),
            Move(Elephant, (7, 8), (9, 6)),
        })

    def test_horse_discover_check_constraint(self):
        board = Xiangqi([
            [None, None, None, None, King(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Horse(False), None, None, None, None],
            [None, None, None, None, Advisor(True), None, None, None, None],
            [None, None, None, King(True), None, None, None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (9, 3), (8, 3)),
            Move(King, (9, 3), (9, 4)),
        })

        board = Xiangqi([
            [None, None, None, None, None, King(False), None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, Horse(False), Rook(True), None, None, None, None, None],
            [None, None, None, Advisor(True), King(True), None, None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(Rook, (8, 3), (8, 2)),
            Move(King, (9, 4), (8, 4)),
            Move(Advisor, (9, 3), (8, 4)),
        })

    def test_pawn_discover_check_constraint(self):
        board = Xiangqi([
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, King(True), None, None, None, None],
            [None, None, None, None, None, Pawn(False), None, None, None],
            [None, None, None, None, None, None, None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (7, 4), (7, 5)),
        })

        board = Xiangqi([
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, Pawn(False), None, None],
            [None, None, None, None, None, King(True), None, None, None],
            [None, None, None, None, None, None, None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (8, 5), (8, 4)),
            Move(King, (8, 5), (9, 5)),
        })

        board = Xiangqi([
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, Pawn(True), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, King(True), None, None, None],
        ], turn=False)
        self.assertEqual(set(board.actions()), set())

        board = Xiangqi([
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Pawn(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, King(True), None, Pawn(False), None, None],
            [None, None, None, None, None, None, None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (8, 4), (9, 4)),
        })

    def test_king_discover_check_constraint(self):
        board = Xiangqi([
            [None, None, None, None, King(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Horse(True), None, None, None, None],
            [None, None, None, None, King(True), None, None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (9, 4), (9, 3)),
            Move(King, (9, 4), (9, 5)),
        })

        board = Xiangqi([
            [None, None, None, None, King(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Horse(False), None, None, None, None],
            [None, None, None, None, King(True), None, None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (9, 4), (9, 3)),
            Move(King, (9, 4), (9, 5)),
        })

        board = Xiangqi([
            [None, None, None, None, King(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Cannon(True), None, None, None, None],
            [None, None, None, None, King(True), None, None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (9, 4), (9, 3)),
            Move(King, (9, 4), (9, 5)),
            Move(Cannon, (8, 4), (7, 4)),
            Move(Cannon, (8, 4), (6, 4)),
            Move(Cannon, (8, 4), (5, 4)),
            Move(Cannon, (8, 4), (4, 4)),
            Move(Cannon, (8, 4), (3, 4)),
            Move(Cannon, (8, 4), (2, 4)),
            Move(Cannon, (8, 4), (1, 4)),
        })

        board = Xiangqi([
            [None, None, None, None, King(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Elephant(True), None, None, None, None],
            [None, None, None, None, Advisor(True), None, None, None, None],
            [None, None, None, None, King(True), None, None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (9, 4), (9, 3)),
            Move(King, (9, 4), (9, 5)),
            Move(Advisor, (8, 4), (9, 3)),
            Move(Advisor, (8, 4), (9, 5)),
            Move(Advisor, (8, 4), (7, 3)),
            Move(Advisor, (8, 4), (7, 5)),
            Move(Elephant, (7, 4), (9, 2)),
            Move(Elephant, (7, 4), (9, 6)),
            Move(Elephant, (7, 4), (5, 2)),
            Move(Elephant, (7, 4), (5, 6)),
        })

    def test_rook_check_constraint(self):
        board = Xiangqi(board=[
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Rook(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, Rook(True), None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, Advisor(True), King(True), Advisor(True), None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(Advisor, (9, 3), (8, 4)),
            Move(Advisor, (9, 5), (8, 4)),
            Move(Rook, (6, 1), (6, 4)),
        })
        
        board = Xiangqi(board=[
            [Rook(True), None, None, None, King(False), None, None, None, None],
            [None, None, None, None, Cannon(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [Rook(False), None, None, Cannon(False), None, King(True), None, None, None],
        ], turn=False)
        self.assertEqual(set(board.actions()), {
            Move(Cannon, (9, 3), (0, 3)),
            Move(Rook, (9, 0), (0, 0)),
        })

        board = Xiangqi(board=[
            [None, None, None, None, None, King(False), None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Cannon(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Elephant(True), None, None, None, None],
            [None, None, None, None, Advisor(True), None, None, None, None],
            [None, None, Rook(False), None, King(True), Advisor(True), None, None, None],
        ])
        self.assertEqual(set(board.actions()), set())

        board = Xiangqi(board=[
            [None, None, None, None, None, King(False), None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Cannon(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Elephant(True), None, None, None, None],
            [None, None, None, None, Advisor(True), None, None, None, None],
            [None, None, None, Rook(False), King(True), Advisor(True), None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (9, 4), (9, 3)),
        })

        board = Xiangqi(board=[
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Cannon(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Elephant(True), None, None, None, None],
            [None, None, None, None, Advisor(True), None, None, None, None],
            [None, None, None, Rook(False), King(True), Advisor(True), None, None, None],
        ])
        self.assertEqual(set(board.actions()), set())

    def test_cannon_check_constraint(self):
        board = Xiangqi(board=[
            [None, None, None, None, King(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Cannon(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, Rook(True), None, None, None, None, None, None, None],
            [None, None, None, None, Elephant(True), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, Advisor(True), King(True), None, None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (9, 4), (9, 5)),
            Move(Advisor, (9, 3), (8, 4)),
            Move(Elephant, (7, 4), (9, 2)),
            Move(Elephant, (7, 4), (9, 6)),
            Move(Elephant, (7, 4), (5, 2)),
            Move(Elephant, (7, 4), (5, 6)),
            Move(Rook, (6, 1), (6, 4)),
        })

        board = Xiangqi(board=[
            [None, None, None, None, King(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, Rook(True), None, None, Cannon(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, Rook(True), None, None, Horse(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, Advisor(True), King(True), None, None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(Rook, (4, 1), (4, 4)),
            Move(Advisor, (9, 3), (8, 4)),
        })

        board = Xiangqi(board=[
            [None, None, None, None, King(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, Rook(True), None, Cannon(False), None, None, None, None],
            [None, None, Rook(True), None, Rook(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, Advisor(True), King(True), Advisor(True), None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(Advisor, (9, 3), (8, 4)),
            Move(Advisor, (9, 5), (8, 4)),
        })

        board = Xiangqi(board=[
            [None, None, None, None, King(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Cannon(False), None, None, None, None],
            [None, None, None, None, Cannon(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, Advisor(True), King(True), Advisor(True), None, None, None],
        ])
        self.assertEqual(set(board.actions()), set())

        board = Xiangqi(board=[
            [None, None, None, None, King(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Pawn(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Cannon(False), None, None, None, None],
            [None, None, None, Rook(True), Cannon(True), Rook(True), None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, Advisor(True), King(True), Advisor(True), None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(Advisor, (9, 5), (8, 4)),
            Move(Advisor, (9, 3), (8, 4)),
            Move(Cannon, (7, 4), (3, 4)),
        })

        board = Xiangqi(board=[
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Cannon(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Rook(False), None, None, None, None],
            [None, None, None, Advisor(True), King(True), Advisor(True), None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (9, 4), (8, 4))
        })

        board = Xiangqi(board=[
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Cannon(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Rook(False), None, None, Rook(False), None],
            [None, None, None, Advisor(True), King(True), Advisor(True), None, None, None],
        ])
        self.assertEqual(set(board.actions()), set())

        board = Xiangqi(board=[
            [None, None, None, None, King(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, Horse(True), None, None, None, None, None, None, None],
            [None, None, None, Cannon(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, Pawn(True), Rook(True), Pawn(True), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, King(True), None, None, None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(Horse, (2, 1), (3, 3)),
            Move(Rook, (6, 3), (3, 3)),
            Move(King, (9, 3), (9, 4)),
        })

    def test_horse_check_constraint(self):
        board = Xiangqi(board=[
            [None, None, None, None, King(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Elephant(True), Horse(False), None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, Advisor(True), King(True), Advisor(True), None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (9, 4), (8, 4)),
        })

        board = Xiangqi(board=[
            [None, None, None, None, King(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Cannon(False), None, None, None, None],
            [None, None, None, None, Horse(True), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, Horse(False), None, None, None, None, None, None],
            [None, None, None, Advisor(True), King(True), Advisor(True), None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(Horse, (6, 4), (8, 3)),
        })

        board = Xiangqi(board=[
            [None, None, None, None, King(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, Cannon(False), None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, Horse(True), None, None, None],
            [None, None, None, None, Advisor(True), None, None, None, None],
            [None, None, None, None, None, King(True), None, Horse(False), Cannon(False)],
        ])
        self.assertEqual(set(board.actions()), {
            Move(Horse, (7, 5), (9, 6)),
        })

    def test_pawn_check_constraint(self):
        board = Xiangqi(board=[
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, Rook(True), None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Pawn(False), None, None, None, None],
            [None, None, None, None, King(True), None, None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (9, 4), (9, 5)),
            Move(King, (9, 4), (8, 4)),
        })

        board = Xiangqi(board=[
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, Rook(True), None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Pawn(False), Pawn(False), None, None, None],
            [None, None, None, None, King(True), None, None, None, None],
        ])
        self.assertEqual(set(board.actions()), set())

    def test_rook_constraining_king(self):
        board = Xiangqi(board=[
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Rook(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, King(True), None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (9, 5), (8, 5)),
        })

        board = Xiangqi(board=[
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, King(True), None, None, None, None],
            [None, None, None, Advisor(True), None, Advisor(True), Rook(False), None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (8, 4), (9, 4)),
            Move(King, (8, 4), (7, 4)),
            Move(King, (8, 4), (8, 5)),
        })

    def test_cannon_constrain_king(self):
        board = Xiangqi(board=[
            [None, None, None, None, King(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Cannon(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Advisor(True), None, None, None, None],
            [None, None, None, King(True), None, Advisor(True), None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (9, 3), (8, 3)),
            Move(Advisor, (8, 4), (7, 3)),
            Move(Advisor, (8, 4), (7, 5)),
        })

        board = Xiangqi(board=[
            [None, None, None, None, King(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Cannon(False), None, None, None, None],
            [None, None, None, None, Pawn(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Advisor(True), None, None, None, None],
            [None, None, None, King(True), None, Advisor(True), None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (9, 3), (9, 4)),
            Move(King, (9, 3), (8, 3)),
            Move(Advisor, (8, 4), (7, 3)),
            Move(Advisor, (8, 4), (7, 5)),
        })

    def test_horse_constrain_king(self):
        board = Xiangqi(board=[
            [None, None, None, None, King(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Cannon(False), None, None, None, None],
            [None, None, None, None, Horse(False), None, None, None, None],
            [None, None, None, None, Elephant(True), Horse(False), None, None, None],
            [None, None, None, None, King(True), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
        ])
        self.assertEqual(set(board.actions()), {
            Move(King, (8, 4), (8, 5)),
            Move(King, (8, 4), (8, 3)),
        })

        board = Xiangqi(board=[
            [None, None, None, None, King(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, Horse(False), None, None, None, None, None, None],
            [None, None, None, King(True), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
        ])
        self.assertEqual(set(board.actions()), set())

    def test_comparison(self):
        board1 = Xiangqi()
        board2 = Xiangqi()
        self.assertEqual(board1, board2)
        self.assertEqual(hash(board1), hash(board2))
    
    def test_parse_board(self):
        board_str = (
            "R1 H1 E1 A1 K1 A1 E1 -- --\n"
            "-- -- -- -- -- -- -- -- R1\n"
            "-- C1 -- -- C1 -- H1 -- --\n"
            "P1 -- P1 -- P1 -- P1 -- P1\n"
            "-- -- -- -- -- -- -- -- --\n"
            "-- -- -- -- -- -- -- -- --\n"
            "P0 -- P0 -- P0 -- P0 -- P0\n"
            "-- C0 -- -- C0 -- H0 -- --\n"
            "-- -- -- -- -- -- -- -- --\n"
            "R0 H0 E0 A0 K0 A0 E0 R0 --"
        )
        board = Xiangqi.from_string(board_str, turn=True)
        expected_board = Xiangqi(board=[
            [Rook(False), Horse(False), Elephant(False), Advisor(False), King(False), Advisor(False), Elephant(False), None, None],
            [None, None, None, None, None, None, None, None, Rook(False)],
            [None, Cannon(False), None, None, Cannon(False), None, Horse(False), None, None],
            [Pawn(False), None, Pawn(False), None, Pawn(False), None, Pawn(False), None, Pawn(False)],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [Pawn(True), None, Pawn(True), None, Pawn(True), None, Pawn(True), None, Pawn(True)],
            [None, Cannon(True), None, None, Cannon(True), None, Horse(True), None, None],
            [None, None, None, None, None, None, None, None, None],
            [Rook(True), Horse(True), Elephant(True), Advisor(True), King(True), Advisor(True), Elephant(True), Rook(True), None],
        ], turn=True)
        self.assertEqual(board, expected_board)

class TestMove(TestCase):
    def test_king_move(self):
        board = Xiangqi(board=[
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, Advisor(True), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, King(True), None, None, None, None],
        ])
        move = board.parse_move('K5.4')
        self.assertEqual(move, Move(King, (9, 4), (9, 5)))
        board = board.move(move)
        move = board.parse_move('K4.5')
        self.assertEqual(move, Move(King, (1, 3), (1, 4)))
        board = board.move(move)
        move = board.parse_move('K4+1')
        self.assertEqual(move, Move(King, (9, 5), (8, 5)))
        board = board.move(move)
        move = board.parse_move('K5-1')
        self.assertEqual(move, Move(King, (1, 4), (0, 4)))

    def test_advisor_move(self):
        board = Xiangqi(board=[
            [None, None, None, Advisor(False), None, King(False), None, None, None],
            [None, None, None, None, Advisor(False), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Advisor(True), None, None, None, None],
            [None, None, None, Advisor(True), King(True), None, None, None, None],
        ])
        move = board.parse_move('A5+4')
        self.assertEqual(move, Move(Advisor, (8, 4), (7, 5)))
        board = board.move(move)
        move = board.parse_move('A5+6')
        self.assertEqual(move, Move(Advisor, (1, 4), (2, 5)))
        board = board.move(move)
        move = board.parse_move('A6+5')
        self.assertEqual(move, Move(Advisor, (9, 3), (8, 4)))
        board = board.move(move)
        move = board.parse_move('A6-5')
        self.assertEqual(move, Move(Advisor, (2, 5), (1, 4)))

    def test_elephant_move(self):
        board = Xiangqi(board=[
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, Elephant(False), None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, King(True), Elephant(True), None, None],
        ])
        move = board.parse_move('E3+5')
        self.assertEqual(move, Move(Elephant, (9, 6), (7, 4)))
        board = board.move(move)
        move = board.parse_move('E7-9')
        self.assertEqual(move, Move(Elephant, (4, 6), (2, 8)))
        board = board.move(move)
        move = board.parse_move('E5+7')
        self.assertEqual(move, Move(Elephant, (7, 4), (5, 2)))
        board = board.move(move)
        move = board.parse_move('E9-7')
        self.assertEqual(move, Move(Elephant, (2, 8), (0, 6)))

    def test_horse_move(self):
        board = Xiangqi(board=[
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, Horse(False), None, None, None],
            [None, None, None, None, None, None, Rook(False), None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, Pawn(True), None],
            [None, None, None, None, None, None, None, None, Horse(True)],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, King(True), None, None, None, None],
        ])
        move = board.parse_move('H1+3')
        self.assertEqual(move, Move(Horse, (5, 8), (4, 6)))
        board = board.move(move)
        move = board.parse_move('H6+8')
        self.assertEqual(move, Move(Horse, (1, 5), (2, 7)))
        board = board.move(move)
        move = board.parse_move('H3+2')
        self.assertEqual(move, Move(Horse, (4, 6), (2, 7)))

    def test_rook_move(self):
        board = Xiangqi(board=[
            [None, None, Cannon(False), None, None, None, None, None, None],
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [Rook(False), None, None, None, None, None, None, None, None],
            [None, Rook(True), None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, King(True), None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
        ])
        move = board.parse_move('R8.2')
        self.assertEqual(move, Move(Rook, (6, 1), (6, 7)))
        board = board.move(move)
        move = board.parse_move('R1+2')
        self.assertEqual(move, Move(Rook, (5, 0), (7, 0)))
        board = board.move(move)
        move = board.parse_move('R2+6')
        self.assertEqual(move, Move(Rook, (6, 7), (0, 7)))
        board = board.move(move)
        move = board.parse_move('R1-7')
        self.assertEqual(move, Move(Rook, (7, 0), (0, 0)))
        board = board.move(move)
        move = board.parse_move('R2.7')
        self.assertEqual(move, Move(Rook, (0, 7), (0, 2)))

    def test_cannon_move(self):
        board = Xiangqi(board=[
            [None, None, Elephant(False), None, King(False), None, Elephant(False), None, None],
            [None, None, Cannon(False), None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, Pawn(False), None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, Pawn(True), Cannon(True), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, Elephant(True), King(True), None, None, None, None, None],
        ])
        move = board.parse_move('C6.3')
        self.assertEqual(move, Move(Cannon, (6, 3), (6, 6)))
        board = board.move(move)
        move = board.parse_move('C3+4')
        self.assertEqual(move, Move(Cannon, (1, 2), (5, 2)))
        board = board.move(move)
        move = board.parse_move('C3+6')
        self.assertEqual(move, Move(Cannon, (6, 6), (0, 6)))
        board = board.move(move)
        move = board.parse_move('C3+4')
        self.assertEqual(move, Move(Cannon, (5, 2), (9, 2)))
        board = board.move(move)
        move = board.parse_move('C3.7')
        self.assertEqual(move, Move(Cannon, (0, 6), (0, 2)))
        board = board.move(move)
        move = board.parse_move('C3-9')
        self.assertEqual(move, Move(Cannon, (9, 2), (0, 2)))

    def test_pawn_move(self):
        board = Xiangqi(board=[
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, Pawn(False), None, None, None, None],
            [None, None, None, None, None, None, Pawn(True), None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, King(True), None, None, None, None],
        ])
        move = board.parse_move('P3+1')
        self.assertEqual(move, Move(Pawn, (6, 6), (5, 6)))
        board = board.move(move)
        move = board.parse_move('P5+1')
        self.assertEqual(move, Move(Pawn, (5, 4), (6, 4)))
        board = board.move(move)
        move = board.parse_move('P3+1')
        self.assertEqual(move, Move(Pawn, (5, 6), (4, 6)))
        board = board.move(move)
        move = board.parse_move('P5.6')
        self.assertEqual(move, Move(Pawn, (6, 4), (6, 5)))
        board = board.move(move)
        move = board.parse_move('P3.2')
        self.assertEqual(move, Move(Pawn, (4, 6), (4, 7)))

    def test_front_back_move(self):
        board = Xiangqi(board=[
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, Cannon(False), None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, Cannon(False), None, None, None, None, None, None],
            [Rook(True), None, None, None, None, None, None, None, None],
            [None, None, Pawn(True), None, None, None, None, None, None],
            [Rook(True), None, None, None, None, None, Pawn(False), None, None],
            [None, None, Horse(True), None, None, None, Pawn(False), None, None],
            [None, None, Horse(True), None, None, None, None, None, None],
            [None, None, None, None, None, King(True), None, None, None],
        ])
        move = board.parse_move('+R+1')
        self.assertEqual(move, Move(Rook, (4, 0), (3, 0)))
        board = board.move(move)
        move = board.parse_move('-C+4')
        self.assertEqual(move, Move(Cannon, (1, 2), (5, 2)))
        board = board.move(move)
        move = board.parse_move('-R-2')
        self.assertEqual(move, Move(Rook, (6, 0), (8, 0)))
        board = board.move(move)
        move = board.parse_move('-C.6')
        self.assertEqual(move, Move(Cannon, (3, 2), (3, 5)))
        board = board.move(move)
        move = board.parse_move('-H+5')
        self.assertEqual(move, Move(Horse, (8, 2), (7, 4)))
        board = board.move(move)
        move = board.parse_move('+P.6')
        self.assertEqual(move, Move(Pawn, (7, 6), (7, 5)))

        board = Xiangqi(board=[
            [None, None, Elephant(False), King(False), None, Advisor(False), None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, Advisor(False), None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, Elephant(False), None, None, None, None, None, None],
            [None, None, Elephant(True), None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, Advisor(True), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, Elephant(True), Advisor(True), None, King(True), None, None, None],
        ])
        move = board.parse_move('+E-5')
        self.assertEqual(move, Move(Elephant, (5, 2), (7, 4)))
        board = board.move(move)
        move = board.parse_move('-E+5')
        self.assertEqual(move, Move(Elephant, (0, 2), (2, 4)))
        board = board.move(move)
        move = board.parse_move('-A+5')
        self.assertEqual(move, Move(Advisor, (9, 3), (8, 4)))
        board = board.move(move)
        move = board.parse_move('+A-5')
        self.assertEqual(move, Move(Advisor, (2, 5), (1, 4)))

    def test_pawn_order_move(self):
        board = Xiangqi(board=[
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, Pawn(True), None, None],
            [None, None, None, None, None, None, Pawn(True), None, None],
            [None, None, Pawn(True), None, None, None, None, None, None],
            [None, None, Pawn(True), None, None, None, None, None, None],
            [None, None, None, None, None, None, Pawn(True), None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, King(True), None, None, None, None],
        ])
        move = board.parse_move('13+1')
        self.assertEqual(move, Move(Pawn, (2, 6), (1, 6)))
        move = board.parse_move('33+1')
        self.assertEqual(move, Move(Pawn, (6, 6), (5, 6)))
        move = board.parse_move('23.4')
        self.assertEqual(move, Move(Pawn, (3, 6), (3, 5)))
        move = board.parse_move('17.8')
        self.assertEqual(move, Move(Pawn, (4, 2), (4, 1)))

    def test_board_str(self):
        board = Xiangqi()
        self.assertEqual(str(board), f'R1 H1 E1 A1 K1 A1 E1 H1 R1' \
                        + '\n-- -- -- -- -- -- -- -- --' \
                        + '\n-- C1 -- -- -- -- -- C1 --' \
                        + '\nP1 -- P1 -- P1 -- P1 -- P1' \
                        + '\n-- -- -- -- -- -- -- -- --' \
                        + '\n-- -- -- -- -- -- -- -- --' \
                        + '\nP0 -- P0 -- P0 -- P0 -- P0' \
                        + '\n-- C0 -- -- -- -- -- C0 --' \
                        + '\n-- -- -- -- -- -- -- -- --' \
                        + '\nR0 H0 E0 A0 K0 A0 E0 H0 R0' \
                        + '\nturn: 0')

    def test_load_board(self):
        with open('board_test.in') as f:
            board_string = f.read()
            board = Xiangqi.from_string(board_string)
        self.assertEqual(board, Xiangqi(board=[
            [Rook(False), None, Elephant(False), Advisor(False), King(False), Advisor(False), Elephant(False), Rook(False), None],
            [None, None, None, None, None, None, None, None, None],
            [None, Cannon(False), Horse(False), None, None, None, Horse(False), Cannon(False), None],
            [Pawn(False), None, Pawn(False), None, Pawn(False), None, None, Rook(True), Pawn(False)],
            [None, None, None, None, None, None, Pawn(False), None, None],
            [None, None, None, None, None, None, None, None, None],
            [Pawn(True), None, Pawn(True), None, Pawn(True), None, Pawn(True), None, Pawn(True)],
            [None, Cannon(True), None, None, Cannon(True), None, Horse(True), None, None],
            [None, None, None, None, None, None, None, None, None],
            [Rook(True), Horse(True), Elephant(True), Advisor(True), King(True), Advisor(True), Elephant(True), None, None],
        ]))

    def test_move_notation_no_duplicate(self):
        board = Xiangqi(board=[
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, Pawn(True), None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [Horse(True), None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, Cannon(True), None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, Rook(True), None],
            [None, None, None, None, King(True), None, None, None, None],
        ])
        move = Move(Horse, (5, 0), (3, 1))
        self.assertEqual(move.to_notation(board), 'H9+8')
        move = Move(Horse, (5, 0), (6, 2))
        self.assertEqual(move.to_notation(board), 'H9-7')
        move = Move(Cannon, (6, 6), (3, 6))
        self.assertEqual(move.to_notation(board), 'C3+3')
        move = Move(Rook, (8, 7), (8, 0))
        self.assertEqual(move.to_notation(board), 'R2.9')
        move = Move(Rook, (8, 7), (9, 7))
        self.assertEqual(move.to_notation(board), 'R2-1')
        move = Move(Pawn, (2, 2), (2, 1))
        self.assertEqual(move.to_notation(board), 'P7.8')

        board = Xiangqi(board=[
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, Cannon(False), None, None, None, None, None, None, None],
            [None, Rook(False), None, None, None, None, None, None, None],
            [None, None, None, None, None, None, Horse(False), None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, Pawn(False), None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, Horse(True), None, None, King(True), None, None, None, None],
        ], turn=False)
        move = Move(Cannon, (2, 1), (9, 1))
        self.assertEqual(move.to_notation(board), 'C2+7')
        move = Move(Cannon, (2, 1), (0, 1))
        self.assertEqual(move.to_notation(board), 'C2-2')
        move = Move(Rook, (3, 1), (3, 4))
        self.assertEqual(move.to_notation(board), 'R2.5')
        move = Move(Horse, (4, 6), (5, 4))
        self.assertEqual(move.to_notation(board), 'H7+5')
        move = Move(Horse, (4, 6), (2, 7))
        self.assertEqual(move.to_notation(board), 'H7-8')
        move = Move(Pawn, (6, 7), (7, 7))
        self.assertEqual(move.to_notation(board), 'P8+1')

    def test_move_notation_duplicate(self):
        board = Xiangqi(board=[
            [None, None, None, King(False), None, None, None, None, None],
            [None, None, None, None, None, None, Horse(True), None, None],
            [None, None, Pawn(True), None, None, None, None, None, None],
            [None, None, None, None, None, None, Horse(True), None, None],
            [None, None, Pawn(True), None, None, None, None, None, None],
            [None, Rook(True), None, None, None, None, Elephant(True), None, None],
            [None, None, Pawn(True), None, None, None, None, None, Cannon(True)],
            [None, Rook(True), None, Advisor(True), None, None, None, None, None],
            [None, None, None, None, None, None, None, None, Cannon(True)],
            [None, None, None, Advisor(True), King(True), None, Elephant(True), None, None],
        ])
        move = Move(Rook, (5, 1), (4, 1))
        self.assertEqual(move.to_notation(board), '+R+1')
        move = Move(Rook, (7, 1), (7, 2))
        self.assertEqual(move.to_notation(board), '-R.7')
        move = Move(Pawn, (2, 2), (1, 2))
        self.assertEqual(move.to_notation(board), '17+1')
        move = Move(Pawn, (4, 2), (4, 3))
        self.assertEqual(move.to_notation(board), '27.6')
        move = Move(Pawn, (6, 2), (5, 2))
        self.assertEqual(move.to_notation(board), '37+1')
        move = Move(Advisor, (7, 3), (8, 4))
        self.assertEqual(move.to_notation(board), '+A-5')
        move = Move(Advisor, (9, 3), (8, 4))
        self.assertEqual(move.to_notation(board), '-A+5')
        move = Move(Elephant, (5, 6), (7, 4))
        self.assertEqual(move.to_notation(board), '+E-5')
        move = Move(Elephant, (9, 6), (7, 8))
        self.assertEqual(move.to_notation(board), '-E+1')
        move = Move(Horse, (1, 6), (0, 8))
        self.assertEqual(move.to_notation(board), '+H+1')
        move = Move(Horse, (3, 6), (1, 5))
        self.assertEqual(move.to_notation(board), '-H+4')
        move = Move(Cannon, (8, 8), (9, 8))
        self.assertEqual(move.to_notation(board), '-C-1')

        board = Xiangqi(board=[
            [None, None, None, King(False), None, Advisor(False), None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, Advisor(False), None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, Pawn(False), None],
            [None, None, None, None, None, None, None, Pawn(False), None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, King(True), None, None, None, None],
        ], turn=False)
        move = Move(Advisor, (0, 5), (1, 4))
        self.assertEqual(move.to_notation(board), '-A+5')
        move = Move(Advisor, (2, 5), (1, 4))
        self.assertEqual(move.to_notation(board), '+A-5')
        move = Move(Pawn, (6, 7), (6, 6))
        self.assertEqual(move.to_notation(board), '-P.7')
        move = Move(Pawn, (7, 7), (8, 7))
        self.assertEqual(move.to_notation(board), '+P+1')
