import random
from copy import deepcopy
from datasets import Dataset

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
    
    def from_string(string: str) -> "TicTacToe":
        lines = string.split("\n")
        board = [[1 if c == "X" else -1 if c == "O" else 0 for c in line.split()] for line in lines if line.strip()]
        return TicTacToe(board)

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

def get_dataset(num_samples: int = 1000):
    def get_prompt(board: TicTacToe) -> str:
        return f"""
You are an expert at playing tictactoe. Your current board is:

{board}

You are playing as {board.get_turn()}.

Determine your best next move as {board.get_turn()}. Explain your answer. At the end, return your move in the format `Final Answer: row, column`

Where row is 0, 1, or 2 counting from top down, and column is 0, 1, or 2 counting from left to right.
"""
    dataset: list[str] = []
    for _ in range(num_samples):
        ttt = TicTacToe.generate_board()
        dataset.append(get_prompt(ttt))
    return Dataset.from_dict({"query": dataset})
