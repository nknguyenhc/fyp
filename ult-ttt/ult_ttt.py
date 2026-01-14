import numpy as np
from dataclasses import dataclass


Action = tuple[int, int, int, int]
LocalBoardAction = tuple[int, int]

def to_cell_str(cell_value: int) -> str:
    if cell_value == 0:
        return '-'
    elif cell_value == 1:
        return 'X'
    elif cell_value == 2:
        return 'O'
    else:
        raise ValueError(f"Invalid cell value: {cell_value}")

def convert_board_to_string(board: np.ndarray) -> str:
    output = []
    horizontal_separator = "------------------------------"
    
    for super_row in range(3):
        for sub_row in range(3):
            line_parts = []
            for super_col in range(3):
                sub_board = board[super_row][super_col]
                sub_line = "  ".join(to_cell_str(sub_board[sub_row][sub_col]) for sub_col in range(3))
                line_parts.append(sub_line)
            full_line = "  | ".join(line_parts)
            output.append(full_line)
        if super_row != 2:
            output.append(horizontal_separator)
    
    return '\n'.join(output)

ENDLINE = '\n'


@dataclass(frozen=True)
class ImmutableState:
    board: np.ndarray
    prev_local_action: LocalBoardAction | None
    fill_num: 1 | 2
    local_board_status: np.ndarray = None

    def __post_init__(self):
        object.__setattr__(self, 'local_board_status', get_local_board_status(self.board))

    def __eq__(self, other):
        return np.all(self.board == other.board) and self.prev_local_action == other.prev_local_action and self.fill_num == other.fill_num

    def __repr__(self):
        return f"""State(
    board=
        {convert_board_to_string(self.board).replace(ENDLINE, ENDLINE+'        ')}, 
    local_board_status=
        {str(self.local_board_status).replace(ENDLINE, ENDLINE+'        ')}, 
    prev_local_action={self.prev_local_action}, 
    fill_num={self.fill_num}
)
"""


def get_local_board_action(action: Action) -> LocalBoardAction:
    meta_row, meta_col, local_row, local_col = action
    return LocalBoardAction((local_row, local_col))


def board_status(board: np.ndarray) -> int:
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != 0:
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] != 0:
            return board[0][i]
    if board[0][0] == board[1][1] == board[2][2] != 0:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != 0:
        return board[0][2]
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                return 0
    return 3


def get_local_board_status(board: np.ndarray) -> None:
    local_board_status: np.ndarray = np.array([[0 for i in range(3)] for j in range(3)])
    for i in range(3):
        for j in range(3):
            local_board_status[i][j] = board_status(board[i][j])
    return local_board_status


def is_valid_action(state: ImmutableState, action: Action) -> bool:
    if not isinstance(action, tuple):
        return False
    if len(action) != 4:
        return False
    i, j, k, l = action
    if type(i) != int or type(j) != int or type(k) != int or type(l) != int:
        return False
    if state.local_board_status[i][j] != 0:
        return False
    if state.board[i][j][k][l] != 0:
        return False
    if state.prev_local_action is None:
        return True
    prev_row, prev_col = state.prev_local_action
    if prev_row == i and prev_col == j:
        return True
    return state.local_board_status[prev_row][prev_col] != 0


def _get_all_valid_free_actions(state: ImmutableState) -> list[Action]:
    valid_actions: list[Action] = []
    for i in range(3):
        for j in range(3):
            if state.local_board_status[i][j] != 0:
                continue
            for k in range(3):
                for l in range(3):
                    if state.board[i][j][k][l] == 0:
                        valid_actions.append((i, j, k, l))
    return valid_actions


def get_all_valid_actions(state: ImmutableState) -> list[Action]:
    if state.prev_local_action is None:
        return _get_all_valid_free_actions(state)
    prev_row, prev_col = state.prev_local_action
    if state.local_board_status[prev_row][prev_col] != 0:
        return _get_all_valid_free_actions(state)
    valid_actions: list[Action] = []
    for i in range(3):
        for j in range(3):
            if state.board[prev_row][prev_col][i][j] == 0:
                valid_actions.append((prev_row, prev_col, i, j))
    return valid_actions


def get_next_turn_fill_num(fill_num):
    return 3 - fill_num


def get_random_valid_action(state: ImmutableState) -> Action:
    valid_actions = get_all_valid_actions(state)
    return valid_actions[np.random.randint(len(valid_actions))]


def change_state(state: ImmutableState, action: Action, check_valid_action = True) -> ImmutableState:
    if check_valid_action:
        assert is_valid_action(state, action), f"Invalid action: {action}"
    i, j, k, l = action
    new_board = state.board.copy()
    new_board[i][j][k][l] = state.fill_num
    new_state = ImmutableState(board=new_board, fill_num=get_next_turn_fill_num(state.fill_num), prev_local_action=get_local_board_action(action))
    return new_state


def is_terminal(state: ImmutableState) -> bool:
    return board_status(state.local_board_status) != 0

def clone_state(state: ImmutableState) -> ImmutableState:
    return ImmutableState(board=state.board.copy(), fill_num=state.fill_num, prev_local_action=state.prev_local_action)

def generate_game_history(num_latest_moves: int = None) -> tuple[list[ImmutableState], list[Action]]:
    states: list[ImmutableState] = []
    actions: list[Action] = []
    state = ImmutableState(board=np.zeros((3, 3, 3, 3)), fill_num=1, prev_local_action=None)
    states.append(clone_state(state))
    n = np.random.randint(5, 15)
    while not is_terminal(state) and len(actions) < n:
        action = get_random_valid_action(state)
        actions.append(action)
        state = change_state(state, action)
        states.append(clone_state(state))
    if num_latest_moves is not None:
        states = states[-(num_latest_moves + 1):]
        actions = actions[-num_latest_moves:]
    return states, actions
