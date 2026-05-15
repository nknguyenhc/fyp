import random
from copy import deepcopy

class TicTacToe:
    def __init__(self, board: list[list[int]], turn: int = 1):
        self.board = board
        self.turn = turn

    def generate_board() -> "TicTacToe":
        num_moves = random.randint(0, 4)
        board = [[0] * 3 for _ in range(3)]
        ttt = TicTacToe(board)
        for i in range(num_moves):
            ttt = ttt.rand_move()
        return ttt
    
    def generate_game_history() -> tuple[list["TicTacToe"], list[tuple[int, int]]]:
        ttt = TicTacToe([[0] * 3 for _ in range(3)])
        history = [ttt]
        moves = []
        num_moves = random.randint(1, 4)
        for _ in range(num_moves):
            free_positions = [(i, j) for i in range(3) for j in range(3) if ttt.board[i][j] == 0]
            assert free_positions, "No more free positions"
            i, j = random.choice(free_positions)
            new_board = deepcopy(ttt.board)
            new_board[i][j] = ttt.turn
            ttt = TicTacToe(new_board, -ttt.turn)
            history.append(ttt)
            moves.append((i, j))
        return history, moves

    def rand_move(self):
        free_positions = [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == 0]
        if not free_positions:
            return self
        i, j = random.choice(free_positions)
        board = deepcopy(self.board)
        board[i][j] = self.turn
        return TicTacToe(board, -self.turn)

    def is_valid_move(self, move: tuple[int, int]) -> bool:
        x, y = move
        return 0 <= x < 3 and 0 <= y < 3 and self.board[x][y] == 0
    
    def _format_number(self, num: int) -> str:
        if num == 1:
            return "X"
        elif num == -1:
            return "O"
        return "-"

    def __repr__(self):
        return f"""
{self._format_number(self.board[0][0])} {self._format_number(self.board[0][1])} {self._format_number(self.board[0][2])}
{self._format_number(self.board[1][0])} {self._format_number(self.board[1][1])} {self._format_number(self.board[1][2])}
{self._format_number(self.board[2][0])} {self._format_number(self.board[2][1])} {self._format_number(self.board[2][2])}
"""
    
    def __str__(self):
        return self.__repr__()

    def get_turn(self) -> str:
        return self._format_number(self.turn)
