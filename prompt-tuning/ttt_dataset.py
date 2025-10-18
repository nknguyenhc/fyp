from datasets import Dataset

from ult_ttt import *

init_prompt = prompt = """You are an expert at playing ultimate tictactoe. The board is 9x9 consisting of 3x3 smaller subboards where each cell can be empty (-), contain an X, or contain an O. To indicate a move, Use the corresponding number for the cell where you want to place your mark:

1  2  3  | 10 11 12 | 19 20 21
4  5  6  | 13 14 15 | 22 23 24
7  8  9  | 16 17 18 | 25 26 27
------------------------------
28 29 30 | 37 38 39 | 46 47 48
31 32 33 | 40 41 42 | 49 50 51
34 35 36 | 43 44 45 | 52 53 54
------------------------------
55 56 57 | 64 65 66 | 73 74 75
58 59 60 | 67 68 69 | 76 77 78
61 62 63 | 70 71 72 | 79 80 81

Note that a player's move determines where the opponent can move in the next move. For example, if player A moves at top-left corner of a subboard, the next player has to move at the top-left subboard. If the board one being sent to, in this example the top-left subboard, is already won by any player or is full, then the player can move anywhere on the board (except from subboards already won or are full). The player making the first move is always free to move anywhere.

The game history is given below. Respond only with the next move by indicating the number corresponding to the cell where you want to place your mark. Do not include any explanations or additional text.

"""

def get_dataset(num_samples: int = 1000):
    def get_prompt(history: tuple[list[ImmutableState], list[Action]]) -> str:
        boards, moves = history
        prompt = f"Board:\n{convert_board_to_string(boards[0].board)}"
        for board, move in zip(boards[1:], moves):
            move_num = 27 * move[0] + 9 * move[1] + 3 * move[2] + move[3] + 1
            prompt += f"\nMove: {move_num}\nBoard:\n{convert_board_to_string(board.board)}"
        prompt += "\nMove: "
        return prompt

    game_histories: list[tuple[list[ImmutableState], list[Action]]] = [generate_game_history() for _ in range(num_samples)]
    dataset: list[str] = [get_prompt(game_history) for game_history in game_histories]
    return Dataset.from_dict({"query": dataset})
