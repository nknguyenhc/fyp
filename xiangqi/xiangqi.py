from copy import deepcopy
from itertools import chain

def manhanttan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

class Xiangqi():
    red_king_positions = set([(i, j) for i in range(7, 10) for j in range(3, 6)])
    black_king_positions = set([(i, j) for i in range(0, 3) for j in range(3, 6)])

    def __init__(self, board=None, turn=True, king_positions=None, copy=True):
        """Instantiates a new board.
        Board is either not given, which means a default board,
        or an array of 10x9, each element is either None or an instance of Piece class.
        If board is given, assume that it is valid.
        Turn is true if it is red player's turn (first), or False if it is black player's turn (second).
        We do deepcopy here to prevent shared usage elsewhere.
        `copy` flag is to indicate whether to deep copy. Turn this to False only if
        this board is readonly, i.e. inputs are not mutated in this instance.
        """
        if board:
            self.board = deepcopy(board) if copy else board
        else:
            self.board = [
                [Rook(False), Horse(False), Elephant(False), Advisor(False), King(False),
                    Advisor(False), Elephant(False), Horse(False), Rook(False)],
                [None, None, None, None, None, None, None, None, None],
                [None, Cannon(False), None, None, None, None, None, Cannon(False), None],
                [Pawn(False), None, Pawn(False), None, Pawn(False), None, Pawn(False),
                    None, Pawn(False)],
                [None, None, None, None, None, None, None, None, None],
                [None, None, None, None, None, None, None, None, None],
                [Pawn(True), None, Pawn(True), None, Pawn(True), None, Pawn(True),
                    None, Pawn(True)],
                [None, Cannon(True), None, None, None, None, None, Cannon(True), None],
                [None, None, None, None, None, None, None, None, None],
                [Rook(True), Horse(True), Elephant(True), Advisor(True), King(True),
                    Advisor(True), Elephant(True), Horse(True), Rook(True)],
            ]
        self.turn = turn
        if king_positions:
            self.king_positions = deepcopy(king_positions) if copy else king_positions
        else:
            self.king_positions = self._find_king_positions()
        self.hash_value = None
        self.constraints = None # cached constraints

    def _find_king_positions(self):
        king_positions = [None, None]
        for i in range(10):
            for j in range(9):
                if self.board[i][j] is None or not isinstance(self.board[i][j], King):
                    continue
                if self.board[i][j].turn:
                    king_positions[0] = (i, j)
                else:
                    king_positions[1] = (i, j)
        return king_positions

    def from_string(board_string, *args, **kwargs):
        """Given a string representation of the board,
        returns the corresponding board.
        """
        def read_piece(piece_string):
            if piece_string == '--':
                return None

            match piece_string[1]:
                case '0':
                    turn = True
                case '1':
                    turn = False
                case _:
                    raise InvalidBoardException()

            match piece_string[0]:
                case 'K':
                    return King(turn)
                case 'A':
                    return Advisor(turn)
                case 'E':
                    return Elephant(turn)
                case 'H':
                    return Horse(turn)
                case 'R':
                    return Rook(turn)
                case 'C':
                    return Cannon(turn)
                case 'P':
                    return Pawn(turn)
                case _:
                    raise InvalidBoardException()

        def read_row(row_string):
            pieces = [read_piece(piece) for piece in row_string.split(' ')]
            if len(pieces) != 9:
                raise InvalidBoardException()
            return pieces

        board = [read_row(row.strip()) for row in board_string.split('\n')]
        if len(board) != 10:
            raise InvalidBoardException()
        return Xiangqi(board=board, *args, **kwargs)

    def actions(self):
        """Returns a list of actions available at this state.
        """
        actions = []
        constraints = self._get_constraints()
        for i, row in enumerate(self.board):
            for j, piece in enumerate(row):
                if piece is None:
                    continue
                for action in piece.actions(self, (i, j)):
                    constraints_satisfied = True
                    for constraint in constraints:
                        if not constraint.satisfies(action):
                            constraints_satisfied = False
                            break
                    if constraints_satisfied:
                        actions.append(action)
        return actions

    def _get_constraints(self):
        if self.constraints is not None:
            return self.constraints
        king_position = self.king_positions[0] if self.turn else self.king_positions[1]
        self.constraints = list(chain.from_iterable([
            self._get_horse_constraints(king_position),
            self._get_cannon_and_rook_constraints(king_position),
            self._get_pawn_constraints(king_position),
            self._get_king_constraints(king_position, self.king_positions[1] if self.turn else self.king_positions[0]),
        ]))
        return self.constraints

    def _get_horse_constraints(self, king_position):
        constraints = []
        def check_position(horse_position, pin_position):
            if not self._is_enemy_piece_type(horse_position, Horse):
                return
            if self.board[pin_position[0]][pin_position[1]] is None:
                constraints.append(HorseCheckConstraint(king_position, horse_position, pin_position))
            elif self.board[pin_position[0]][pin_position[1]].turn == self.turn:
                constraints.append(HorseDiscoverCheckConstraint(king_position, horse_position, pin_position))

        if king_position[0] + 2 <= 9:
            if king_position[1] + 1 <= 8:
                check_position((king_position[0] + 2, king_position[1] + 1), (king_position[0] + 1, king_position[1] + 1))
            if king_position[1] - 1 >= 0:
                check_position((king_position[0] + 2, king_position[1] - 1), (king_position[0] + 1, king_position[1] - 1))
        if king_position[0] - 2 >= 0:
            if king_position[1] + 1 <= 8:
                check_position((king_position[0] - 2, king_position[1] + 1), (king_position[0] - 1, king_position[1] + 1))
            if king_position[1] - 1 >= 0:
                check_position((king_position[0] - 2, king_position[1] - 1), (king_position[0] - 1, king_position[1] - 1))
        if king_position[1] + 2 <= 8:
            if king_position[0] + 1 <= 9:
                check_position((king_position[0] + 1, king_position[1] + 2), (king_position[0] + 1, king_position[1] + 1))
            if king_position[0] - 1 >= 0:
                check_position((king_position[0] - 1, king_position[1] + 2), (king_position[0] - 1, king_position[1] + 1))
        if king_position[1] - 2 >= 0:
            if king_position[0] + 1 <= 9:
                check_position((king_position[0] + 1, king_position[1] - 2), (king_position[0] + 1, king_position[1] - 1))
            if king_position[0] - 1 >= 0:
                check_position((king_position[0] - 1, king_position[1] - 2), (king_position[0] - 1, king_position[1] - 1))

        return constraints

    def _get_cannon_and_rook_constraints(self, king_position):
        constraints = []
        def consider_cell(row, col, piece_positions):
            target_position = (row, col)
            match len(piece_positions):
                case 0:
                    if self._is_enemy_piece_type(target_position, Rook):
                        constraints.append(RookCheckConstraint(king_position, target_position))
                    elif self._is_enemy_piece_type(target_position, Cannon):
                        constraints.append(CannonDiscoverNoPieceCheckConstraint(king_position, target_position))
                case 1:
                    if self._is_enemy_piece_type(target_position, Rook):
                        constraints.append(RookDiscoverCheckConstraint(king_position, target_position, piece_positions[0]))
                    elif self._is_enemy_piece_type(target_position, Cannon):
                        constraints.append(CannonCheckConstraint(king_position, target_position, piece_positions[0]))
                case 2:
                    if self._is_enemy_piece_type(target_position, Cannon):
                        constraints.append(CannonDiscoverTwoPiecesCheckConstraint(king_position, target_position, tuple(piece_positions)))
            if self.board[row][col] is not None:
                piece_positions.append(target_position)

        def consider_column(row_range, col):
            piece_positions = []
            for row in row_range:
                consider_cell(row, col, piece_positions)
                if len(piece_positions) > 2:
                    break

        def consider_row(col_range, row):
            piece_positions = []
            for col in col_range:
                consider_cell(row, col, piece_positions)
                if len(piece_positions) > 2:
                    break

        consider_column(range(king_position[0] + 1, 10), king_position[1])
        consider_column(range(king_position[0] - 1, -1 , -1), king_position[1])
        consider_row(range(king_position[1] + 1, 9), king_position[0])
        consider_row(range(king_position[1] - 1, -1, -1), king_position[0])
        return constraints

    def _get_pawn_constraints(self, king_position):
        constraints = []
        if self._is_enemy_piece_type((king_position[0], king_position[1] + 1), Pawn):
            constraints.append(PawnCheckConstraint(king_position, (king_position[0], king_position[1] + 1)))
        if self._is_enemy_piece_type((king_position[0], king_position[1] - 1), Pawn):
            constraints.append(PawnCheckConstraint(king_position, (king_position[0], king_position[1] - 1)))

        if self.turn:
            row = king_position[0] - 1
        else:
            row = king_position[0] + 1
        if row >= 0 and row <= 9 and self._is_enemy_piece_type((row, king_position[1]), Pawn):
            constraints.append(PawnCheckConstraint(king_position, (row, king_position[1])))

        return constraints

    def _get_king_constraints(self, king_position, enemy_king_position):
        if king_position[1] != enemy_king_position[1]:
            return []
        min_row = min(king_position[0], enemy_king_position[0])
        max_row = max(king_position[0], enemy_king_position[0])
        col = king_position[1]
        piece_position = None
        for i in range(min_row + 1, max_row):
            if self.board[i][col] is None:
                continue
            if piece_position is not None:
                return []
            piece_position = (i, col)
        return [KingDiscoverCheckConstraint(king_position, piece_position)]

    def _is_enemy_piece_type(self, position, piecetype):
        """Checks if the position on the board contains an enemy piece of the specified type.
        """
        return isinstance(self.board[position[0]][position[1]], piecetype) and self.board[position[0]][position[1]].turn != self.turn

    def _can_move_to(self, position):
        """Checks if the position is available that another piece can move to.
        Assuming that the given position is a valid position in the board.
        This means either the slot is empty, or the slot is occupied by a piece of the opposite side.
        """
        piece = self.board[position[0]][position[1]]
        if piece is None:
            return True
        return piece.turn != self.turn

    def move(self, action: "Move") -> "Xiangqi":
        """Returns a new board from application of given action on this board.
        Assuming that the action is a valid action.
        """
        next_state = Xiangqi(board=self.board, turn=not self.turn,
            king_positions=self.king_positions)
        next_state.board[action.to_coords[0]][action.to_coords[1]] = next_state.board[
            action.from_coords[0]][action.from_coords[1]]
        next_state.board[action.from_coords[0]][action.from_coords[1]] = None
        if action.piecetype == King:
            if self.turn:
                next_state.king_positions[0] = action.to_coords
            else:
                next_state.king_positions[1] = action.to_coords
        # print(action)
        # print(next_state)
        # print(f"{self.king_positions=}")
        # print(f"{next_state.king_positions=}")
        return next_state

    def parse_move(self, move_string):
        """Parses a move string into a legitimate.
        Throws InvalidMoveException if the move is invalid.
        """
        if len(move_string) != 4:
            raise InvalidMoveException(move_string)

        match move_string[0]:
            case 'K':
                move = self._parse_king_move(move_string)
            case 'A':
                move = self._parse_advisor_move(move_string)
            case 'E':
                move = self._parse_elephant_move(move_string)
            case 'H':
                move = self._parse_horse_move(move_string)
            case 'R':
                move = self._parse_rook_move(move_string)
            case 'C':
                move = self._parse_cannon_move(move_string)
            case 'P':
                move = self._parse_pawn_move(move_string)
            case '+' | '-':
                move = self._parse_piece_with_order_move(move_string)
            case '1' | '2' | '3' | '4' | '5':
                move = self._parse_pawn_with_order_move(move_string)
            case _:
                raise InvalidMoveException(move_string)

        possible_moves = self.actions()
        if move not in possible_moves:
            raise InvalidMoveException(move_string)
        return move

    def _is_friendly_piece_type(self, position, piecetype):
        """Checks if the piece at the given position is a friendly piece of the given type.
        """
        return isinstance(self.board[position[0]][position[1]], piecetype) and self.board[position[0]][position[1]].turn == self.turn

    def _from_col(self, move_string):
        """Returns the starting col index from the move string.
        """
        try:
            return 9 - int(move_string[1]) if self.turn else int(move_string[1]) - 1
        except ValueError:
            raise InvalidMoveException(move_string)

    def _dest_col(self, move_string):
        """Returns the destination col from the move string.
        """
        try:
            return 9 - int(move_string[3]) if self.turn else int(move_string[3]) - 1
        except ValueError:
            raise InvalidMoveException(move_string)

    def _dest_row(self, curr_row, row_increment):
        """Returns the row index from the move string.
        """
        return curr_row - row_increment if self.turn else curr_row + row_increment

    def _get_row_difference(self, move_string):
        try:
            return int(move_string[3])
        except ValueError:
            raise InvalidMoveException(move_string)

    def _search_col(self, col, piecetype, index=0, max_count=1):
        """Search the column for the piece type.
        col: the column index to search for
        piecetype: the piece type to search for
        index: the index of the piecetype in the column, 0-indexed
        max_count: the maximum count of the piecetype in the column
        Returns the row index of the target piece, or None if not found.
        """
        count = 0
        curr_row = -1
        for row in (range(0, 10) if self.turn else range(9, -1, -1)):
            if not self._is_friendly_piece_type((row, col), piecetype):
                continue
            if count == max_count:
                return None
            if count == index:
                curr_row = row
            count += 1

        if max_count > 1:
            if count < 2:
                return None
        return curr_row if curr_row >= 0 else None

    def _parse_king_move(self, move_string):
        col = self._from_col(move_string)
        for row in ([7, 8, 9] if self.turn else [0, 1, 2]):
            if not self._is_friendly_piece_type((row, col), King):
                continue

            match move_string[2]:
                case '+':
                    if row == 7 or row == 2:
                        raise InvalidMoveException(move_string)
                    if move_string[3] != '1':
                        raise InvalidMoveException(move_string)
                    dest_row = self._dest_row(row, 1)
                    return Move(King, (row, col), (dest_row, col))
                case '-':
                    if row == 9 or row == 0:
                        raise InvalidMoveException(move_string)
                    if move_string[3] != '1':
                        raise InvalidMoveException(move_string)
                    dest_row = self._dest_row(row, -1)
                    return Move(King, (row, col), (dest_row, col))
                case '.':
                    dest_col = self._dest_col(move_string)
                    if abs(dest_col - col) != 1 or dest_col < 3 or dest_col > 5:
                        raise InvalidMoveException(move_string)
                    return Move(King, (row, col), (row, dest_col))
                case _:
                    raise InvalidMoveException(move_string)

        raise InvalidMoveException(move_string)

    def _parse_advisor_move(self, move_string):
        col = self._from_col(move_string)
        match col:
            case 3 | 5:
                match move_string[2]:
                    case '+':
                        row = 7 if self.turn else 2
                        if self._is_friendly_piece_type((row, col), Advisor):
                            raise InvalidMoveException(move_string)
                        row = 9 if self.turn else 0
                        if not self._is_friendly_piece_type((row, col), Advisor):
                            raise InvalidMoveException(move_string)
                        target_row = self._dest_row(row, 1)
                    case '-':
                        row = 9 if self.turn else 0
                        if self._is_friendly_piece_type((row, col), Advisor):
                            raise InvalidMoveException(move_string)
                        row = 7 if self.turn else 2
                        if not self._is_friendly_piece_type((row, col), Advisor):
                            raise InvalidMoveException(move_string)
                        target_row = self._dest_row(row, -1)
                    case _:
                        raise InvalidMoveException(move_string)
                if move_string[3] != '5':
                    raise InvalidMoveException(move_string)
                return Move(Advisor, (row, col), (target_row, 4))
            case 4:
                row = 8 if self.turn else 1
                if not self._is_friendly_piece_type((row, col), Advisor):
                    raise InvalidMoveException(move_string)
                target_col = self._dest_col(move_string)
                if target_col != 3 and target_col != 5:
                    raise InvalidMoveException(move_string)
                match move_string[2]:
                    case '+':
                        target_row = self._dest_row(row, 1)
                    case '-':
                        target_row = self._dest_row(row, -1)
                    case _:
                        raise InvalidMoveException(move_string)
                return Move(Advisor, (row, col), (target_row, target_col))
            case _:
                raise InvalidMoveException(move_string)

    def _parse_elephant_move(self, move_string):
        col = self._from_col(move_string)
        target_col = self._dest_col(move_string)
        match col:
            case 0 | 4 | 8:
                row = 7 if self.turn else 2
                if not self._is_friendly_piece_type((row, col), Elephant):
                    raise InvalidMoveException(move_string)
                if abs(target_col - col) != 2:
                    raise InvalidMoveException(move_string)
                match move_string[2]:
                    case '+':
                        target_row = self._dest_row(row, 2)
                    case '-':
                        target_row = self._dest_row(row, -2)
                    case _:
                        raise InvalidMoveException(move_string)
                if self.board[(target_row + row) // 2][(target_col + col) // 2] is not None:
                    raise InvalidMoveException(move_string)
                return Move(Elephant, (row, col), (target_row, target_col))
            case 2 | 6:
                target_row = 7 if self.turn else 2
                match move_string[2]:
                    case '+':
                        row = 5 if self.turn else 4
                        if self._is_friendly_piece_type((row, col), Elephant):
                            raise InvalidMoveException(move_string)
                        row = 9 if self.turn else 0
                        if not self._is_friendly_piece_type((row, col), Elephant):
                            raise InvalidMoveException(move_string)
                    case '-':
                        row = 9 if self.turn else 0
                        if self._is_friendly_piece_type((row, col), Elephant):
                            raise InvalidMoveException(move_string)
                        row = 5 if self.turn else 4
                        if not self._is_friendly_piece_type((row, col), Elephant):
                            raise InvalidMoveException(move_string)
                    case _:
                        raise InvalidMoveException(move_string)
                return Move(Elephant, (row, col), (target_row, target_col))
            case _:
                raise InvalidMoveException(move_string)

    def _parse_horse_move(self, move_string, col=None, row=None):
        col = col if col != None else self._from_col(move_string)
        row = row if row != None else self._search_col(col, Horse)
        if row is None:
            raise InvalidMoveException(move_string)

        target_col = self._dest_col(move_string)
        match abs(target_col - col):
            case 1:
                match move_string[2]:
                    case '+':
                        target_row = self._dest_row(row, 2)
                    case '-':
                        target_row = self._dest_row(row, -2)
                    case _:
                        raise InvalidMoveException(move_string)
            case 2:
                match move_string[2]:
                    case '+':
                        target_row = self._dest_row(row, 1)
                    case '-':
                        target_row = self._dest_row(row, -1)
                    case _:
                        raise InvalidMoveException(move_string)
            case _:
                raise InvalidMoveException(move_string)
        if target_row < 0 or target_row >= 10:
            raise InvalidMoveException(move_string)
        return Move(Horse, (row, col), (target_row, target_col))

    def _parse_rook_move(self, move_string, col=None, row=None):
        col = col if col != None else self._from_col(move_string)
        row = row if row != None else self._search_col(col, Rook)
        if row is None:
            raise InvalidMoveException(move_string)

        match move_string[2]:
            case '+':
                row_increment = self._get_row_difference(move_string)
                target_row = self._dest_row(row, row_increment)
                if target_row > 9 and target_row < 0:
                    raise InvalidMoveException(move_string)
                for test_row in (range(row - 1, target_row, -1) if self.turn else range(row + 1, target_row)):
                    if self.board[test_row][col] is not None:
                        raise InvalidMoveException(move_string)
                return Move(Rook, (row, col), (target_row, col))
            case '-':
                row_decrement = self._get_row_difference(move_string)
                target_row = self._dest_row(row, -row_decrement)
                if target_row > 9 or target_row < 0:
                    raise InvalidMoveException(move_string)
                for test_row in (range(row + 1, target_row) if self.turn else range(row - 1, target_row, -1)):
                    if self.board[test_row][col] is not None:
                        raise InvalidMoveException(move_string)
                return Move(Rook, (row, col), (target_row, col))
            case '.':
                target_col = self._dest_col(move_string)
                if col < 0 or col > 8 or target_col == col:
                    raise InvalidMoveException(move_string)
                for test_col in (range(col + 1, target_col) if target_col > col else range(col - 1, target_col, -1)):
                    if self.board[row][test_col] is not None:
                        raise InvalidMoveException(move_string)
                return Move(Rook, (row, col), (row, target_col))
            case _:
                raise InvalidMoveException(move_string)

    def _parse_cannon_move(self, move_string, col=None, row=None):
        col = col if col != None else self._from_col(move_string)
        row = row if row != None else self._search_col(col, Cannon)
        if row is None:
            raise InvalidMoveException(move_string)

        def test_col(row, col, row_range, target_row):
            if target_row > 9 or target_row < 0 or target_row == row:
                raise InvalidMoveException(move_string)
            piece_count = 0
            for row in row_range:
                if self.board[row][col] is None:
                    continue
                piece_count += 1
                if piece_count >= 2:
                    raise InvalidMoveException(move_string)
            if piece_count == 0:
                if self.board[target_row][col] is not None:
                    raise InvalidMoveException(move_string)
            elif piece_count == 1:
                if self.board[target_row][col] is None:
                    raise InvalidMoveException(move_string)

        def test_row(row, col, col_range, target_col):
            if target_col > 8 or target_col < 0 or target_col == col:
                raise InvalidMoveException(move_string)
            piece_count = 0
            for col in col_range:
                if self.board[row][col] is None:
                    continue
                piece_count += 1
                if piece_count >= 2:
                    raise InvalidMoveException(move_string)
            if piece_count == 0:
                if self.board[row][target_col] is not None:
                    raise InvalidMoveException(move_string)
            elif piece_count == 1:
                if self.board[row][target_col] is None:
                    raise InvalidMoveException(move_string)

        match move_string[2]:
            case '+':
                row_increment = self._get_row_difference(move_string)
                target_row = self._dest_row(row, row_increment)
                test_col(row, col, range(row - 1, target_row, -1) if self.turn else range(row + 1, target_row), target_row)
                return Move(Cannon, (row, col), (target_row, col))
            case '-':
                row_decrement = self._get_row_difference(move_string)
                target_row = self._dest_row(row, -row_decrement)
                test_col(row, col, range(row + 1, target_row) if self.turn else range(row - 1, target_row, -1), target_row)
                return Move(Cannon, (row, col), (target_row, col))
            case '.':
                target_col = self._dest_col(move_string)
                test_row(row, col, range(col + 1, target_col) if target_col > col else range(col - 1, target_col, -1), target_col)
                return Move(Cannon, (row, col), (row, target_col))
            case _:
                raise InvalidMoveException(move_string)

    def _parse_pawn_move(self, move_string, col=None, row=None):
        col = col if col != None else self._from_col(move_string)
        row = row if row != None else self._search_col(col, Pawn)
        if row is None:
            raise InvalidMoveException(move_string)

        match move_string[2]:
            case '+':
                if row == 0 or row == 9:
                    raise InvalidMoveException(move_string)
                if move_string[3] != '1':
                    raise InvalidMoveException(move_string)
                target_row = self._dest_row(row, 1)
                return Move(Pawn, (row, col), (target_row, col))
            case '.':
                if self.turn:
                    if row > 4:
                        raise InvalidMoveException(move_string)
                else:
                    if row < 5:
                        raise InvalidMoveException(move_string)
                target_col = self._dest_col(move_string)
                if abs(target_col - col) != 1:
                    raise InvalidMoveException(move_string)
                return Move(Pawn, (row, col), (row, target_col))
            case _:
                raise InvalidMoveException(move_string)

    def _parse_piece_with_order_move(self, move_string):
        match move_string[0]:
            case '+':
                index = 0
            case '-':
                index = 1

        def find_piece(piece_type):
            for col in range(9):
                row = self._search_col(col, piece_type, index=index, max_count=2)
                if row is not None:
                    return row, col
            raise InvalidMoveException(move_string)

        match move_string[1]:
            case 'A':
                return self._parse_advisor_front_back_move(move_string)
            case 'E':
                return self._parse_elephant_front_back_move(move_string)
            case 'H':
                row, col = find_piece(Horse)
                return self._parse_horse_move(move_string, col=col, row=row)
            case 'R':
                row, col = find_piece(Rook)
                return self._parse_rook_move(move_string, col=col, row=row)
            case 'C':
                row, col = find_piece(Cannon)
                return self._parse_cannon_move(move_string, col=col, row=row)
            case 'P':
                row, col = find_piece(Pawn)
                return self._parse_pawn_move(move_string, col=col, row=row)
            case _:
                raise InvalidMoveException(move_string)

    def _parse_advisor_front_back_move(self, move_string):
        target_row = 8 if self.turn else 1
        match move_string[0]:
            case '+':
                if move_string[2:4] != '-5':
                    raise InvalidMoveException(move_string)
                row, other_row = (7, 9) if self.turn else (2, 0)
            case '-':
                if move_string[2:4] != '+5':
                    raise InvalidMoveException(move_string)
                row, other_row = (9, 7) if self.turn else (0, 2)

        for col in [3, 5]:
            if not self._is_friendly_piece_type((other_row, col), Advisor):
                continue
            if not self._is_friendly_piece_type((row, col), Advisor):
                continue
            return Move(Advisor, (row, col), (target_row, 4))
        raise InvalidMoveException(move_string)

    def _parse_elephant_front_back_move(self, move_string):
        target_row = 7 if self.turn else 2
        match move_string[0]:
            case '+':
                row, other_row = (5, 9) if self.turn else (4, 0)
            case '-':
                row, other_row = (9, 5) if self.turn else (0, 4)
        target_col = self._dest_col(move_string)

        for col in [2, 6]:
            if not self._is_friendly_piece_type((other_row, col), Elephant):
                continue
            if not self._is_friendly_piece_type((row, col), Elephant):
                continue
            return Move(Elephant, (row, col), (target_row, target_col))
        raise InvalidMoveException(move_string)

    def _parse_pawn_with_order_move(self, move_string):
        index = int(move_string[0]) - 1
        col = self._from_col(move_string)
        row = self._search_col(col, Pawn, index=index, max_count=5)
        if row is None:
            raise InvalidMoveException(move_string)

        return self._parse_pawn_move(move_string, col=col, row=row)

    def __str__(self):
        def condense_row(row):
            return ' '.join([str(piece) if piece else '--' for piece in row])

        return '\n'.join([condense_row(row) for row in self.board]) + f'\nturn: {0 if self.turn else 1}'

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, Xiangqi):
            return False
        return self.board == other.board and self.turn == other.turn
    
    def __hash__(self):
        if self.hash_value is not None:
            return self.hash_value
        
        board_tuple = tuple([tuple(row) for row in self.board])
        self.hash_value = hash(board_tuple)
        return self.hash_value
    
    def __eq__(self, other):
        if not isinstance(other, Xiangqi):
            return False
        
        return self.turn == other.turn and self.board == other.board


class InvalidBoardException(Exception):
    def __init__(self):
        super().__init__("The content of the given file does not represent a true board.")


class InvalidMoveException(Exception):
    def __init__(self, move_string):
        super().__init__(f"Invalid move string: \"{move_string}\" is not a valid move")


class CheckConstraint:
    def __init__(self, king_position):
        """Represents a constraint where pieces cannot discover a check against the current player,
        or where pieces have to respond to a check from the opponent.
        """
        self.king_position = king_position

    def satisfies(self, move: "Move"):
        """Determine if a move violates this constraint.
        Subclasses should override this method.
        """
        raise NotImplementedError

    def is_check(self):
        """Returns if this constraint represents a piece checking the king,
        i.e. the opponent has to respond to the check.
        """
        raise NotImplementedError

    def __repr__(self):
        return self.__str__()


class RookDiscoverCheckConstraint(CheckConstraint):
    def __init__(self, king_position, rook_position, piece_position):
        """Instantiates a constraint that the only piece in between the king and an opponent rook cannot move away.
        Assuming that king_position and rook_position has either same x- or y-coordinates.
        """
        super().__init__(king_position)
        self.rook_position = rook_position
        self.piece_position = piece_position

    def satisfies(self, move: "Move"):
        if move.from_coords == self.king_position and move.to_coords == self.piece_position:
            return False
        if move.from_coords != self.piece_position:
            return True
        if self.king_position[0] == self.rook_position[0]:
            if move.to_coords[0] != self.king_position[0]:
                return False
            return (self.king_position[1] < move.to_coords[1] and move.to_coords[1] <= self.rook_position[1]) \
                or (self.king_position[1] > move.to_coords[1] and move.to_coords[1] >= self.rook_position[1])
        else:
            if move.to_coords[1] != self.king_position[1]:
                return False
            return (self.king_position[0] < move.to_coords[0] and move.to_coords[0] <= self.rook_position[0]) \
                or (self.king_position[0] > move.to_coords[0] and move.to_coords[0] >= self.rook_position[0])

    def is_check(self):
        return False

    def __str__(self):
        return f"<R-dconstraint, K: {self.king_position}, R: {self.rook_position}, p: {self.piece_position}>"


class CannonDiscoverTwoPiecesCheckConstraint(CheckConstraint):
    def __init__(self, king_position, cannon_position, piece_positions):
        """Instantiates a constraint that the two only pieces in between the king and an opponent cannon cannot move away.
        piece_positions is a tuple of two elements, each representing the coordinates of a piece in between.
        Assuming that the king_position, cannon_position and piece_positions have either same x- or y-coordinates.
        """
        super().__init__(king_position)
        self.cannon_position = cannon_position
        self.piece_positions = piece_positions

    def satisfies(self, move: "Move"):
        if move.from_coords == self.king_position and move.to_coords in self.piece_positions:
            return False
        if move.from_coords not in self.piece_positions:
            return True
        if move.to_coords in self.piece_positions:
            return False
        if self.king_position[0] == self.cannon_position[0]:
            if move.to_coords[0] != self.king_position[0]:
                return False
            return (self.king_position[1] < move.to_coords[1] and move.to_coords[1] <= self.cannon_position[1]) \
                or (self.king_position[1] > move.to_coords[1] and move.to_coords[1] >= self.cannon_position[1])
        else:
            if move.to_coords[1] != self.king_position[1]:
                return False
            return (self.king_position[0] < move.to_coords[0] and move.to_coords[0] <= self.cannon_position[0]) \
                or (self.king_position[0] > move.to_coords[0] and move.to_coords[0] >= self.cannon_position[0])

    def is_check(self):
        return False

    def __str__(self):
        return f"<C2p-dconstraint, K: {self.king_position}, C: {self.cannon_position}, p: {self.piece_positions}>"


class CannonDiscoverNoPieceCheckConstraint(CheckConstraint):
    def __init__(self, king_position, cannon_position):
        """Instantiates a constraint if an opponent cannon is facing the king directly, no piece can enter in between.
        Assuming that king_position and cannon_position have the same x- or y-coordinates.
        """
        super().__init__(king_position)
        self.cannon_position = cannon_position

    def satisfies(self, move: "Move"):
        if self.king_position[0] == self.cannon_position[0]:
            if move.to_coords[0] != self.king_position[0]:
                return True
            if move.from_coords == self.king_position:
                return True
            return (move.to_coords[1] < self.king_position[1] and move.to_coords[1] <= self.cannon_position[1]) \
                or (move.to_coords[1] > self.king_position[1] and move.to_coords[1] >= self.cannon_position[1])
        else:
            if move.to_coords[1] != self.king_position[1]:
                return True
            if move.from_coords == self.king_position:
                return True
            return (move.to_coords[0] < self.king_position[0] and move.to_coords[0] <= self.cannon_position[0]) \
                or (move.to_coords[0] > self.king_position[0] and move.to_coords[0] >= self.cannon_position[0])

    def is_check(self):
        return False

    def __str__(self):
        return f"<C0p-dconstraint, K: {self.king_position}, C: {self.cannon_position}>"


class HorseDiscoverCheckConstraint(CheckConstraint):
    def __init__(self, king_position, horse_position, piece_position):
        """Instantiates a constraint that a piece blocking the horse check cannot move away.
        """
        super().__init__(king_position)
        self.horse_position = horse_position
        self.piece_position = piece_position

    def satisfies(self, move: "Move"):
        if move.from_coords != self.piece_position:
            return True
        return move.to_coords == self.horse_position

    def is_check(self):
        return False

    def __str__(self):
        return f"<H-dconstraint, K: {self.king_position}, H: {self.horse_position}, p: {self.piece_position}>"


class KingDiscoverCheckConstraint(CheckConstraint):
    def __init__(self, king_position, piece_position):
        """Instantiates a constraint that the only piece standing in between the kings cannot move away.
        """
        super().__init__(king_position)
        self.piece_position = piece_position

    def satisfies(self, move: "Move"):
        if move.from_coords == self.king_position:
            return move.to_coords != self.piece_position
        if move.from_coords != self.piece_position:
            return True
        return move.to_coords[1] == self.piece_position[1]

    def is_check(self):
        return False

    def __str__(self):
        return f"<K-dconstraint, K: {self.king_position}, p: {self.piece_position}>"


class RookCheckConstraint(CheckConstraint):
    def __init__(self, king_position, rook_position):
        """Instantiates a constraint that a move must respond to an opponent rook checking the king.
        Assuming that king_position and rook_position has the same x- or y-coordinates.
        """
        super().__init__(self)
        self.king_position = king_position
        self.rook_position = rook_position

    def satisfies(self, move: "Move"):
        if move.to_coords == self.rook_position:
            return True
        if self.king_position[0] == self.rook_position[0]:
            if move.from_coords == self.king_position:
                return move.to_coords[0] != self.king_position[0] or move.to_coords == self.rook_position
            if move.to_coords[0] != self.king_position[0]:
                return False
            return (self.king_position[1] < move.to_coords[1] and move.to_coords[1] <= self.rook_position[1]) \
                or (self.king_position[1] > move.to_coords[1] and move.to_coords[1] >= self.rook_position[1])
        else:
            if move.from_coords == self.king_position:
                return move.to_coords[1] != self.king_position[1] or move.to_coords == self.rook_position
            if move.to_coords[1] != self.king_position[1]:
                return False
            return (self.king_position[0] < move.to_coords[0] and move.to_coords[0] <= self.rook_position[0]) \
                or (self.king_position[0] > move.to_coords[0] and move.to_coords[0] >= self.rook_position[0])

    def is_check(self):
        return True

    def __str__(self):
        return f"<R-constraint, K: {self.king_position}, R: {self.rook_position}>"


class CannonCheckConstraint(CheckConstraint):
    def __init__(self, king_position, cannon_position, piece_position):
        """Instantiates a constraint that a move must respond to an opponent cannon checking the king.
        Assuming that king_position, cannon_position and piece_position has the same x- or y-coordinates.
        """
        super().__init__(king_position)
        self.cannon_position = cannon_position
        self.piece_position = piece_position

    def satisfies(self, move: "Move"):
        if move.to_coords == self.cannon_position:
            return True
        if self.king_position[0] == self.cannon_position[0]:
            if move.from_coords == self.king_position:
                return move.to_coords[0] != self.king_position[0] or move.to_coords[1] == self.piece_position[1]
            if move.from_coords == self.piece_position:
                if move.to_coords[0] != self.king_position[0]:
                    return True
                if self.king_position[1] < self.cannon_position[1]:
                    return move.to_coords[1] < self.king_position[1] or move.to_coords[1] > self.cannon_position[1]
                else:
                    return move.to_coords[1] > self.king_position[1] or move.to_coords[1] < self.cannon_position[1]
            if move.to_coords[0] != self.king_position[0]:
                return False
            if move.to_coords[1] == self.piece_position[1]:
                return False
            return (self.king_position[1] < move.to_coords[1] and move.to_coords[1] <= self.cannon_position[1]) \
                or (self.king_position[1] > move.to_coords[1] and move.to_coords[1] >= self.cannon_position[1])
        else:
            if move.from_coords == self.king_position:
                return move.to_coords[1] != self.king_position[1] or move.to_coords[0] == self.piece_position[0]
            if move.from_coords == self.piece_position:
                if move.to_coords[1] != self.king_position[1]:
                    return True
                if self.king_position[0] < self.cannon_position[0]:
                    return move.to_coords[0] < self.king_position[0] or move.to_coords[0] > self.cannon_position[0]
                else:
                    return move.to_coords[0] > self.king_position[0] or move.to_coords[0] < self.cannon_position[0]
            if move.to_coords[1] != self.king_position[1]:
                return False
            if move.to_coords[0] == self.piece_position[0]:
                return False
            return (self.king_position[0] < move.to_coords[0] and move.to_coords[0] <= self.cannon_position[0]) \
                or (self.king_position[0] > move.to_coords[0] and move.to_coords[0] >= self.cannon_position[0])

    def is_check(self):
        return True

    def __str__(self):
        return f"<C-constraint, K: {self.king_position}, C: {self.cannon_position}, p: {self.piece_position}>"


class HorseCheckConstraint(CheckConstraint):
    def __init__(self, king_position, horse_position, pin_position):
        """Instantiates a constraint that a move must respond to an opponent horse checking the king.
        pin_position is the position current blank on the board where a piece moving to can block the check.
        """
        super().__init__(king_position)
        self.horse_position = horse_position
        self.pin_position = pin_position

    def satisfies(self, move: "Move"):
        return move.from_coords == self.king_position or move.to_coords == self.horse_position \
            or move.to_coords == self.pin_position

    def is_check(self):
        return True

    def __str__(self):
        return f"<H-constraint, K: {self.king_position}, H: {self.horse_position}, pin: {self.pin_position}>"


class PawnCheckConstraint(CheckConstraint):
    def __init__(self, king_position, pawn_position):
        """Instantiates a constraint that a move must respond to an opponent panw checking th king.
        """
        super().__init__(king_position)
        self.pawn_position = pawn_position

    def satisfies(self, move: "Move"):
        return move.from_coords == self.king_position or move.to_coords == self.pawn_position

    def is_check(self):
        return True

    def __str__(self):
        return f"<P-constraint, K: {self.king_position}, P: {self.pawn_position}>"


class MoveMode:
    """Defines the categories of a move.
    Each move must belong to at least 1 category.
    """
    UNDEFINED = 0
    QUIET = 1
    CAPTURE = 2
    CHECK = 4
    ALL = 7


class Move:
    def __init__(self, piecetype, from_coords, to_coords):
        """Instantiates a new move object.
        piecetype is the class associated with the piece.
        from_coords is a tuple of two numbers representing the starting position of the piece (array indices)
        to_coords is a tuple of two numbers representing the ending position of the piece (array indices)
        """
        self.piecetype = piecetype
        self.from_coords = from_coords
        self.to_coords = to_coords

    def __hash__(self):
        return hash((self.piecetype, self.from_coords, self.to_coords))

    def __eq__(self, other):
        if not isinstance(other, Move):
            return False
        return self.piecetype == other.piecetype and self.from_coords == other.from_coords \
            and self.to_coords == other.to_coords

    def __str__(self):
        return f"{self.piecetype.to_string()} {self.from_coords} {self.to_coords}"

    def __repr__(self):
        return self.__str__()

    def to_notation(self, xiangqi):
        """Returns the notation of this move on the board.
        """
        first_half = self.obtain_first_half(xiangqi)
        second_half = self.obtain_second_half(xiangqi)
        return first_half + second_half

    def obtain_first_half(self, xiangqi):
        piece_count, rank = self.piece_count(xiangqi, self.from_coords[1])
        match piece_count:
            case 1:
                return self.piecetype.to_string() + str(self.col_index_to_num(self.from_coords[1], xiangqi.turn))
            case 2:
                return self.rank_to_sign(rank, xiangqi.turn) + self.piecetype.to_string()
            case _:
                return str(self.rank_to_num(piece_count, rank, xiangqi.turn)) + str(self.col_index_to_num(self.from_coords[1], xiangqi.turn))

    def obtain_second_half(self, xiangqi):
        if self.to_coords[0] == self.from_coords[0]:
            return '.' + str(self.col_index_to_num(self.to_coords[1], xiangqi.turn))
        elif self.to_coords[0] > self.from_coords[0]:
            sign = '-' if xiangqi.turn else '+'
        else:
            sign = '+' if xiangqi.turn else '-'

        if self.to_coords[1] == self.from_coords[1]:
            return sign + str(abs(self.to_coords[0] - self.from_coords[0]))
        else:
            return sign + str(self.col_index_to_num(self.to_coords[1], xiangqi.turn))

    def piece_count(self, xiangqi, col):
        piece_count = 0
        for row in range(10):
            if not xiangqi._is_friendly_piece_type((row, col), self.piecetype):
                continue
            piece_count += 1
            if row == self.from_coords[0]:
                rank = piece_count
        return piece_count, rank

    def col_index_to_num(self, col, turn):
        return 9 - col if turn else col + 1

    def rank_to_sign(self, rank, turn):
        if rank == 1:
            return '+' if turn else '-'
        else: # rank == 2
            return '-' if turn else '+'

    def rank_to_num(self, piece_count, rank, turn):
        if turn:
            return rank
        else:
            return piece_count - rank + 1


class Piece:
    def __init__(self, turn):
        """Instantiates a new piece.
        turn determines whether this piece belongs to red player (first player).
        This is an abstract class.
        """
        self.turn = turn

    def actions(self, xiangqi, position) -> list[Move]:
        """Returns the list of actions possible for this piece,
        given the state and its current position on the board.
        Assuming that the position given is a valid board position of this piece.
        Subclasses should override this method.
        """
        raise NotImplementedError

    def to_string():
        """Returns the string representation of the class,
        not associated with any particular instance.
        Subclasses should override this method.
        """
        raise NotImplementedError

    def __str__(self):
        return f"{self.__class__.to_string()}{0 if self.turn else 1}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.turn == other.turn
    
    def _are_constraints_satisfied(self, origin, dest, constraints):
        return all([constraint.is_check() or constraint.satisfies(Move(self.__class__, origin, dest)) for constraint in constraints])
    
    def __hash__(self):
        """Each piece must be hashable.
        """
        raise NotImplementedError
    
    def __eq__(self, other):
        """Each piece must be comparable.
        """
        raise NotImplementedError


class King(Piece):
    safety_malus = (
        (0, 0, 0, 20, 0, 20, 0, 0, 0),
        (0, 0, 0, 150, 300, 100, 0, 0, 0),
        (0, 0, 0, 400, 400, 400, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, -400, -400, -400, 0, 0, 0),
        (0, 0, 0, -150, -300, -150, 0, 0, 0),
        (0, 0, 0, -20, 0, -20, 0, 0, 0),
    )

    def __init__(self, turn):
        super().__init__(turn)

    def actions(self, xiangqi, position):
        if xiangqi.turn != self.turn:
            return []

        actions = []
        dest = (position[0] + 1, position[1])
        if dest[0] <= (9 if self.turn else 2) and xiangqi._can_move_to(dest) \
                and not self._is_move_exposing_check(xiangqi, dest, False):
            actions.append(Move(King, position, dest))
        dest = (position[0] - 1, position[1])
        if dest[0] >= (7 if self.turn else 0) and xiangqi._can_move_to(dest) \
                and not self._is_move_exposing_check(xiangqi, dest, False):
            actions.append(Move(King, position, dest))
        dest = (position[0], position[1] + 1)
        if dest[1] <= 5 and xiangqi._can_move_to(dest) \
                and not self._is_move_exposing_kings(position[1] + 1, xiangqi, position[0]) \
                and not self._is_move_exposing_check(xiangqi, dest, True):
            actions.append(Move(King, position, dest))
        dest = (position[0], position[1] - 1)
        if dest[1] >= 3 and xiangqi._can_move_to(dest) \
                and not self._is_move_exposing_kings(position[1] - 1, xiangqi, position[0]) \
                and not self._is_move_exposing_check(xiangqi, dest, True):
            actions.append(Move(King, position, dest))
        return actions

    def _is_move_exposing_check(self, xiangqi, dest, horizontal):
        """Checks if the move to the destination exposes any check.
        horizontal indicates the move is horizontal.
        """
        return self._is_move_exposing_horse_check(xiangqi, dest) or self._is_move_exposing_pawn_check(xiangqi, dest) \
            or self._is_move_exposing_cannon_or_rook_check(xiangqi, dest, horizontal)

    def _is_move_exposing_horse_check(self, xiangqi, dest):
        # no need to check for dest[1], because this is king move
        if dest[0] > 0 and xiangqi.board[dest[0] - 1][dest[1] - 1] is None:
            if dest[0] > 1 and xiangqi._is_enemy_piece_type((dest[0] - 2, dest[1] - 1), Horse):
                return True
            if xiangqi._is_enemy_piece_type((dest[0] - 1, dest[1] - 2), Horse):
                return True
        if dest[0] > 0 and xiangqi.board[dest[0] - 1][dest[1] + 1] is None:
            if dest[0] > 1 and xiangqi._is_enemy_piece_type((dest[0] - 2, dest[1] + 1), Horse):
                return True
            if xiangqi._is_enemy_piece_type((dest[0] - 1, dest[1] + 2), Horse):
                return True
        if dest[0] < 9 and xiangqi.board[dest[0] + 1][dest[1] - 1] is None:
            if dest[0] < 8 and xiangqi._is_enemy_piece_type((dest[0] + 2, dest[1] - 1), Horse):
                return True
            if xiangqi._is_enemy_piece_type((dest[0] + 1, dest[1] - 2), Horse):
                return True
        if dest[0] < 9 and xiangqi.board[dest[0] + 1][dest[1] + 1] is None:
            if dest[0] < 8 and xiangqi._is_enemy_piece_type((dest[0] + 2, dest[1] + 1), Horse):
                return True
            if xiangqi._is_enemy_piece_type((dest[0] + 1, dest[1] + 2), Horse):
                return True
        return False

    def _is_move_exposing_pawn_check(self, xiangqi, dest):
        if xiangqi._is_enemy_piece_type((dest[0], dest[1] - 1), Pawn):
            return True
        if xiangqi._is_enemy_piece_type((dest[0], dest[1] + 1), Pawn):
            return True
        if self.turn:
            return xiangqi._is_enemy_piece_type((dest[0] - 1, dest[1]), Pawn)
        else:
            return xiangqi._is_enemy_piece_type((dest[0] + 1, dest[1]), Pawn)

    def _is_move_exposing_cannon_or_rook_check(self, xiangqi, dest, horizontal):
        def consider_cell(row, col, piece_count):
            if xiangqi.board[row][col] is None:
                return (piece_count, False)
            target_position = (row, col)
            match piece_count:
                case 0:
                    if xiangqi._is_enemy_piece_type(target_position, Rook):
                        return (1, True)
                    return (1, False)
                case 1:
                    if xiangqi._is_enemy_piece_type(target_position, Cannon):
                        return (2, True)
                    return (2, False)

        def consider_column(col, row_range):
            piece_count, is_check_found = 0, False
            for row in row_range:
                piece_count, is_check_found = consider_cell(row, col, piece_count)
                if is_check_found:
                    return True
                if piece_count == 2:
                    return False
            return False

        def consider_row(row, col_range):
            piece_count, is_check_found = 0, False
            for col in col_range:
                piece_count, is_check_found = consider_cell(row, col, piece_count)
                if is_check_found:
                    return True
                if piece_count == 2:
                    return False
            return False

        if horizontal:
            return consider_column(dest[1], range(dest[0] + 1, 10)) or consider_column(dest[1], range(dest[0] - 1, -1, -1))
        else:
            return consider_row(dest[0], range(dest[1] + 1, 9)) or consider_row(dest[0], range(dest[1] - 1, -1, -1))

    def _is_move_exposing_kings(self, col_index, xiangqi, from_row):
        """Checks if the column with given index in the state has any piece blocking in between,
        so that this King piece can move sideway to that column.
        Returns True if the move would expose the two kings.
        """
        for i in (range(from_row - 1, -1, -1) if self.turn else range(from_row + 1, 10)):
            if xiangqi.board[i][col_index] is None:
                continue
            if isinstance(xiangqi.board[i][col_index], King):
                return True
            else:
                return False
        return False

    def to_string():
        return 'K'

    def __hash__(self):
        return 1
    
    def __eq__(self, other):
        if not isinstance(other, King):
            return False
        
        return self.turn == other.turn


class Advisor(Piece):
    def __init__(self, turn):
        super().__init__(turn)

    def actions(self, xiangqi, position):
        if xiangqi.turn != self.turn:
            return []

        if self.turn:
            match position:
                case (9, 3) | (9, 5) | (7, 3) | (7, 5):
                    if xiangqi._can_move_to((8, 4)):
                        return [Move(Advisor, position, (8, 4))]
                    else:
                        return []
                case (8, 4):
                    actions = []
                    for dest in [(9, 3), (9, 5), (7, 3), (7, 5)]:
                        if xiangqi._can_move_to(dest):
                            actions.append(Move(Advisor, position, dest))
                    return actions
        else:
            match position:
                case (0, 3) | (0, 5) | (2, 3) | (2, 5):
                    if xiangqi._can_move_to((1, 4)):
                        return [Move(Advisor, position, (1, 4))]
                    else:
                        return []
                case (1, 4):
                    actions = []
                    for dest in [(0, 3), (0, 5), (2, 3), (2, 5)]:
                        if xiangqi._can_move_to(dest):
                            actions.append(Move(Advisor, position, dest))
                    return actions

    def to_string():
        return 'A'
    
    def __hash__(self):
        return 2
    
    def __eq__(self, other):
        if not isinstance(other, Advisor):
            return False
        
        return self.turn == other.turn


class Elephant(Piece):
    def __init__(self, turn):
        super().__init__(turn)

    def actions(self, xiangqi, position):
        if xiangqi.turn != self.turn:
            return []

        if self.turn:
            min_row = 5
            max_row = 9
        else:
            min_row = 0
            max_row = 4

        actions = []
        if position[0] + 2 <= max_row and position[1] + 2 <= 8 and xiangqi.board[position[0] + 1][position[1] + 1] is None \
                and xiangqi._can_move_to((position[0] + 2, position[1] + 2)):
            actions.append(Move(Elephant, position, (position[0] + 2, position[1] + 2)))
        if position[0] + 2 <= max_row and position[1] - 2 >= 0 and xiangqi.board[position[0] + 1][position[1] - 1] is None \
                and xiangqi._can_move_to((position[0] + 2, position[1] - 2)):
            actions.append(Move(Elephant, position, (position[0] + 2, position[1] - 2)))
        if position[0] - 2 >= min_row and position[1] + 2 <= 8 and xiangqi.board[position[0] - 1][position[1] + 1] is None \
                and xiangqi._can_move_to((position[0] - 2, position[1] + 2)):
            actions.append(Move(Elephant, position, (position[0] - 2, position[1] + 2)))
        if position[0] - 2 >= min_row and position[1] - 2 >= 0 and xiangqi.board[position[0] - 1][position[1] - 1] is None \
                and xiangqi._can_move_to((position[0] - 2, position[1] - 2)):
            actions.append(Move(Elephant, position, (position[0] - 2, position[1] - 2)))
        return actions

    def to_string():
        return 'E'
    
    def __hash__(self):
        return 4
    
    def __eq__(self, other):
        if not isinstance(other, Elephant):
            return False
        
        return self.turn == other.turn


class Horse(Piece):
    def __init__(self, turn):
        super().__init__(turn)

    def actions(self, xiangqi, position) -> list[Move]:
        if xiangqi.turn != self.turn:
            return []

        actions = []
        if position[0] + 2 <= 9 and xiangqi.board[position[0] + 1][position[1]] is None:
            if position[1] + 1 <= 8 and xiangqi._can_move_to((position[0] + 2, position[1] + 1)):
                actions.append(Move(Horse, position, (position[0] + 2, position[1] + 1)))
            if position[1] - 1 >= 0 and xiangqi._can_move_to((position[0] + 2, position[1] - 1)):
                actions.append(Move(Horse, position, (position[0] + 2, position[1] - 1)))
        if position[0] - 2 >= 0 and xiangqi.board[position[0] - 1][position[1]] is None:
            if position[1] + 1 <= 8 and xiangqi._can_move_to((position[0] - 2, position[1] + 1)):
                actions.append(Move(Horse, position, (position[0] - 2, position[1] + 1)))
            if position[1] - 1 >= 0 and xiangqi._can_move_to((position[0] - 2, position[1] - 1)):
                actions.append(Move(Horse, position, (position[0] - 2, position[1] - 1)))
        if position[1] + 2 <= 8 and xiangqi.board[position[0]][position[1] + 1] is None:
            if position[0] + 1 <= 9 and xiangqi._can_move_to((position[0] + 1, position[1] + 2)):
                actions.append(Move(Horse, position, (position[0] + 1, position[1] + 2)))
            if position[0] - 1 >= 0 and xiangqi._can_move_to((position[0] - 1, position[1] + 2)):
                actions.append(Move(Horse, position, (position[0] - 1, position[1] + 2)))
        if position[1] - 2 >= 0 and xiangqi.board[position[0]][position[1] - 1] is None:
            if position[0] + 1 <= 9 and xiangqi._can_move_to((position[0] + 1, position[1] - 2)):
                actions.append(Move(Horse, position, (position[0] + 1, position[1] - 2)))
            if position[0] - 1 >= 0 and xiangqi._can_move_to((position[0] - 1, position[1] - 2)):
                actions.append(Move(Horse, position, (position[0] - 1, position[1] - 2)))
        return actions

    def to_string():
        return 'H'
    
    def __hash__(self):
        return 8
    
    def __eq__(self, other):
        if not isinstance(other, Horse):
            return False
        
        return self.turn == other.turn


class Rook(Piece):
    def __init__(self, turn):
        super().__init__(turn)

    def actions(self, xiangqi, position, vertical=True, horizontal=True) -> list[Move]:
        if xiangqi.turn != self.turn:
            return []

        actions = []
        if vertical:
            for i in range(position[0] + 1, 10):
                if xiangqi.board[i][position[1]] is None:
                    actions.append(Move(Rook, position, (i, position[1])))
                    continue
                if xiangqi.board[i][position[1]].turn != self.turn:
                    actions.append(Move(Rook, position, (i, position[1])))
                break
            for i in range(position[0] - 1, -1, -1):
                if xiangqi.board[i][position[1]] is None:
                    actions.append(Move(Rook, position, (i, position[1])))
                    continue
                if xiangqi.board[i][position[1]].turn != self.turn:
                    actions.append(Move(Rook, position, (i, position[1])))
                break
        if horizontal:
            for j in range(position[1] + 1, 9):
                if xiangqi.board[position[0]][j] is None:
                    actions.append(Move(Rook, position, (position[0], j)))
                    continue
                if xiangqi.board[position[0]][j].turn != self.turn:
                    actions.append(Move(Rook, position, (position[0], j)))
                break
            for j in range(position[1] - 1, -1, -1):
                if xiangqi.board[position[0]][j] is None:
                    actions.append(Move(Rook, position, (position[0], j)))
                    continue
                if xiangqi.board[position[0]][j].turn != self.turn:
                    actions.append(Move(Rook, position, (position[0], j)))
                break
        return actions

    def to_string():
        return 'R'

    def __hash__(self):
        return 16
    
    def __eq__(self, other):
        if not isinstance(other, Rook):
            return False
        
        return self.turn == other.turn


class Cannon(Piece):
    def __init__(self, turn):
        super().__init__(turn)

    def actions(self, xiangqi, position) -> list[Move]:
        if xiangqi.turn != self.turn:
            return []

        actions = []
        for i in range(position[0] + 1, 10):
            if xiangqi.board[i][position[1]] is None:
                actions.append(Move(Cannon, position, (i, position[1])))
                continue
            for ii in range(i + 1, 10):
                if xiangqi.board[ii][position[1]] is not None:
                    if xiangqi.board[ii][position[1]].turn != self.turn:
                        actions.append(Move(Cannon, position, (ii, position[1])))
                    break
            break
        for i in range(position[0] - 1, -1, -1):
            if xiangqi.board[i][position[1]] is None:
                actions.append(Move(Cannon, position, (i, position[1])))
                continue
            for ii in range(i - 1, -1, -1):
                if xiangqi.board[ii][position[1]] is not None:
                    if xiangqi.board[ii][position[1]].turn != self.turn:
                        actions.append(Move(Cannon, position, (ii, position[1])))
                    break
            break
        for j in range(position[1] + 1, 9):
            if xiangqi.board[position[0]][j] is None:
                actions.append(Move(Cannon, position, (position[0], j)))
                continue
            for jj in range(j + 1, 9):
                if xiangqi.board[position[0]][jj] is not None:
                    if xiangqi.board[position[0]][jj].turn != self.turn:
                        actions.append(Move(Cannon, position, (position[0], jj)))
                    break
            break
        for j in range(position[1] - 1, -1, -1):
            if xiangqi.board[position[0]][j] is None:
                actions.append(Move(Cannon, position, (position[0], j)))
                continue
            for jj in range(j - 1, -1, -1):
                if xiangqi.board[position[0]][jj] is not None:
                    if xiangqi.board[position[0]][jj].turn != self.turn:
                        actions.append(Move(Cannon, position, (position[0], jj)))
                    break
            break
        return actions

    def to_string():
        return 'C'
    
    def __hash__(self):
        return 32
    
    def __eq__(self, other):
        if not isinstance(other, Cannon):
            return False
        
        return self.turn == other.turn


class Pawn(Piece):
    def __init__(self, turn):
        super().__init__(turn)

    def actions(self, xiangqi, position):
        if xiangqi.turn != self.turn:
            return []

        actions = []
        if self.turn:
            if position[0] > 0 and xiangqi._can_move_to((position[0] - 1, position[1])):
                actions.append(Move(Pawn, position, (position[0] - 1, position[1])))
            if position[0] <= 4:
                if position[1] > 0 and xiangqi._can_move_to((position[0], position[1] - 1)):
                    actions.append(Move(Pawn, position, (position[0], position[1] - 1)))
                if position[1] < 8 and xiangqi._can_move_to((position[0], position[1] + 1)):
                    actions.append(Move(Pawn, position, (position[0], position[1] + 1)))
        else:
            if position[0] < 9 and xiangqi._can_move_to((position[0] + 1, position[1])):
                actions.append(Move(Pawn, position, (position[0] + 1, position[1])))
            if position[0] >= 5:
                if position[1] > 0 and xiangqi._can_move_to((position[0], position[1] - 1)):
                    actions.append(Move(Pawn, position, (position[0], position[1] - 1)))
                if position[1] < 8 and xiangqi._can_move_to((position[0], position[1] + 1)):
                    actions.append(Move(Pawn, position, (position[0], position[1] + 1)))
        return actions

    def to_string():
        return 'P'
        
    def __hash__(self):
        return 64
    
    def __eq__(self, other):
        if not isinstance(other, Pawn):
            return False
        
        return self.turn == other.turn
