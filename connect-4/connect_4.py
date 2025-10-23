from typing import List, Tuple
import os
import random

from config import height, width, connect, steal

class State:
    UNDETERMINED = 0
    X = 1
    O = 2
    DRAW = 3

class InvalidBoardStringException(Exception):
    """Representing an exception raised when an invalid string is provided
    in the factory method of `Board` class.
    """
    def __init__(self, message: str):
        """Instantiates a new exception,
        raised when an invalid string is provided to create a new `Board`.
        """
        super().__init__(message)

class Board:
    def _get_winning_lines() -> List[List[List[int]]]:
        directions = [
            (0, 1), (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1), (-1, 0), (-1, 1),
        ]
        winning_lines = [[None for i in range(width)] for j in range(height)]
        for i in range(height):
            for j in range(width):
                lines = []
                for direction in directions:
                    i_ = i + direction[0] * (connect - 1)
                    j_ = j + direction[1] * (connect - 1)
                    if i_ < 0 or i_ >= height or j_ < 0 or j_ >= width:
                        continue
                    line = 0
                    for k in range(connect):
                        line += 2 ** ((i + direction[0] * k) * width + (j + direction[1] * k))
                    lines.append(line)
                winning_lines[i][j] = lines
        return winning_lines

    WINNING_LINES = _get_winning_lines()
    DEFAULT_ACTIONS = [i for i in range(width)]

    def __init__(self, is_X_turn: bool=True,
                 X_table: int=None, O_table: int=None,
                 last_move: Tuple[int, int]=None,
                 move_count: int=0,
                 actions: List[int]=DEFAULT_ACTIONS,
                 ):
        """Instantiates a new table.
        Either:
        1. Both `X_table` and `O_table` are left blank, or
        2. Both `X_table` and `O_table` are provided with the correct dimension.
        """
        self.is_X_turn = is_X_turn
        self.move_count = move_count
        if X_table is not None and O_table is not None:
            self.X_table = X_table
            self.O_table = O_table
            if last_move is None:
                self.winner = State.UNDETERMINED
            else:
                self._determine_winner(last_move)
        else:
            self.X_table = 0
            self.O_table = 0
            self.winner = State.UNDETERMINED
        if actions is not None:
            self._actions = actions
        else:
            self._actions = self._find_actions()
    
    def actions(self) -> List[int]:
        """Returns the set of possible actions in this state.
        Each action is an int indicating the column to move at.
        """
        if steal and self.move_count == 1:
            return self._actions + [-1]
        else:
            return self._actions
    
    def _find_actions(self) -> List[int]:
        """Returns the set of possible actions in this state.
        Each action is an int indicating the column to move at.
        """
        assert os.environ.get("APP_TESTING") == "True"
        actions: List[int] = []
        for col in range(width):
            if self._is_column_movable(col):
                actions.append(col)
        if self.move_count == 1 and steal:
            actions.append(-1)
        return actions
    
    def is_valid_action(self, col: int) -> bool:
        """Determines if the action is valid.
        Only to be used when interacting with the user.
        """
        return col in self._actions

    def _is_column_movable(self, col: int) -> bool:
        """Determines if a piece can be added at the column.
        """
        return not (self.X_table >> ((height - 1) * width + col) & 1) \
            and not (self.O_table >> ((height - 1) * width + col) & 1)
    
    def _determine_winner(self, last_move: Tuple[int, int]) -> None:
        """Assigns to `self.winner` the correct winner at this state.
        """
        if self._is_winner(self.X_table, last_move):
            self.winner = State.X
        elif self._is_winner(self.O_table, last_move):
            self.winner = State.O
        elif self._is_terminal():
            self.winner = State.DRAW
        else:
            self.winner = State.UNDETERMINED
    
    def _is_winner(self, arr: int, last_move: Tuple[int, int]) -> bool:
        """Checks in the following directions at each cell:
        1. Rightwards
        2. Right-downwards
        3. Downwards
        4. Left-downwards
        """
        for winning_line in Board.WINNING_LINES[last_move[0]][last_move[1]]:
            if arr & winning_line == winning_line:
                return True
        return False
    
    def _is_terminal(self) -> bool:
        return self.move_count == height * width
    
    def move(self, col: int) -> "Board":
        """Makes a move at the indicated column.
        Returns a new instance of `Board`.
        """
        if col == -1:
            assert self.move_count == 1 and steal and not self.is_X_turn
            return Board(is_X_turn=True, X_table=0, O_table=self.X_table, move_count=2, actions=self._actions)
        for row in range(height):
            if (self.X_table >> (row * width + col) & 1) \
                or (self.O_table >> (row * width + col) & 1):
                continue
            next_actions = self._actions
            if row == height - 1:
                next_actions = next_actions.copy()
                next_actions.remove(col)
            if self.is_X_turn:
                return Board(
                    is_X_turn=not self.is_X_turn,
                    X_table=self.X_table + 2 ** (row * width + col),
                    O_table=self.O_table,
                    last_move=(row, col),
                    move_count=self.move_count + 1,
                    actions=next_actions,
                )
            else:
                return Board(
                    is_X_turn=not self.is_X_turn,
                    X_table=self.X_table,
                    O_table=self.O_table + 2 ** (row * width + col),
                    last_move=(row, col),
                    move_count=self.move_count + 1,
                    actions=next_actions,
                )
        
        assert False, "Invalid move!"
    
    def __repr__(self):
        board_string = ""
        for row in range(height - 1, -1, -1):
            row_string = ""
            for col in range(width):
                assert not (self.X_table >> (row * width + col) & 1) \
                    or not (self.O_table >> (row * width + col) & 1)
                if (self.X_table >> (row * width + col) & 1):
                    row_string += "X "
                elif (self.O_table >> (row * width + col) & 1):
                    row_string += "O "
                else:
                    row_string += "- "
            board_string += row_string + "\n"
        return board_string

    def __str__(self):
        return self.__repr__()
    
    def from_string(string: str) -> "Board":
        """Returns a board represented by the string.
        The string is the same as the string representation of the board,
        but combined into one line, lines are separated by two space characters.
        """
        parts = string.split('|')
        if len(parts) != 2:
            raise InvalidBoardStringException("Incorrect number of parts by |, expect 2")

        match parts[1]:
            case "T":
                is_X_turn = True
            case "F":
                is_X_turn = False
            case _:
                raise InvalidBoardStringException(f"Incorrect turn token \"{parts[1]}\"")
        
        string = parts[0]

        items = string.split('  ')
        if len(items) != height:
            raise InvalidBoardStringException("Incorrect number of lines")
        
        X_table: List[Tuple[bool, ...]] = []
        O_table: List[Tuple[bool, ...]] = []
        move_count: int = 0
        for row in items:
            X_row: List[bool] = []
            O_row: List[bool] = []
            cells = row.split(' ')
            if len(cells) != width:
                raise InvalidBoardStringException(f"Incorrect number of items in a line: {row}")
            
            for cell in cells:
                match cell:
                    case "X":
                        X_row.append(True)
                        O_row.append(False)
                        move_count += 1
                    case "O":
                        X_row.append(False)
                        O_row.append(True)
                        move_count += 1
                    case "-":
                        X_row.append(False)
                        O_row.append(False)
                    case _:
                        raise InvalidBoardStringException(f"Invalid character \"{cell}\"")
            
            X_table.append(tuple(X_row))
            O_table.append(tuple(O_row))
        
        X_int = 0
        O_int = 0
        for j in range(height):
            for k in range(width):
                if X_table[j][k]:
                    X_int += 2 ** (j * width + k)
                if O_table[j][k]:
                    O_int += 2 ** (j * width + k)
        
        return Board(is_X_turn=is_X_turn,
                     X_table=X_int,
                     O_table=O_int,
                     move_count=move_count,
                     actions=None)
    
    def to_compact_string(self) -> str:
        board_string = ""
        for row in range(height):
            row_string = ""
            for col in range(width):
                if (self.X_table >> (row * width + col) & 1):
                    row_string += "X"
                elif (self.O_table >> (row * width + col) & 1):
                    row_string += "O"
                else:
                    row_string += "-"
                if col != width - 1:
                    row_string += " "
            board_string += row_string
            if row != height - 1:
                board_string += "  "
        if self.is_X_turn:
            board_string += "|T"
        else:
            board_string += "|F"
        return board_string
    
    def __eq__(self, other):
        if not isinstance(other, Board):
            return False
        return self.is_X_turn == other.is_X_turn and \
            self.X_table == other.X_table and \
            self.O_table == other.O_table

def generate_game_history() -> tuple[list[Board], list[int]]:
    boards: list[Board] = [Board()]
    moves: list[int] = []
    current_board = boards[0]
    num_steps = random.randint(3, 9)
    for _ in range(num_steps):
        if current_board.winner != State.UNDETERMINED:
            break
        possible_actions = current_board.actions()
        action = possible_actions[random.randint(0, len(possible_actions) - 1)]
        current_board = current_board.move(action)
        boards.append(current_board)
        moves.append(action)
    if current_board.winner != State.UNDETERMINED:
        return generate_game_history()
    return (boards, moves)
