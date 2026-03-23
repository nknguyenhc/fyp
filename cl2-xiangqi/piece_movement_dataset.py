from datasets import Dataset
import random

from xiangqi import Xiangqi

def _generate_game() -> Xiangqi:
    num_moves = random.randint(10, 30)
    board = Xiangqi()
    for _ in range(num_moves):
        possible_moves = board.actions()
        if not possible_moves:
            break
        move = random.choice(possible_moves)
        board = board.move(move)
    if len(board.actions()) == 0:
        return _generate_game()
    return board

def _get_prompt(board: Xiangqi, num_samples: int) -> list[str]:
    moves = board.actions()
    moves = random.sample(moves, min(num_samples, len(moves)))
    prompts: list[str] = []
    for move in moves:
        prompt = f"""You are an expert at playing Chinese Chess. The board is 10x9 where each cell can be empty (--), contain a red piece (_0), or contain a black piece (_1). The board position numbers are as follows:

 1  2  3  4  5  6  7  8  9
10 11 12 13 14 15 16 17 18
19 20 21 22 23 24 25 26 27
28 29 30 31 32 33 34 35 36
37 38 39 40 41 42 43 44 45
46 47 48 49 50 51 52 53 54
55 56 57 58 59 60 61 62 63
64 65 66 67 68 69 70 71 72
73 74 75 76 77 78 79 80 81
82 83 84 85 86 87 88 89 90

The pieces are represented as follows:
- King: K0 (Red), K1 (Black)
- Advisor: A0 (Red), A1 (Black)
- Elephant: E0 (Red), E1 (Black)
- Horse: H0 (Red), H1 (Black)
- Rook: R0 (Red), R1 (Black)
- Cannon: C0 (Red), C1 (Black)
- Pawn: P0 (Red), P1 (Black)

The red pieces are initially at the bottom half of the board, while the black pieces are initially at the top half of the board. The goal of the game is to checkmate the opponent's king, while protecting your own king. You are currently playing as {"Red" if board.turn else "Black"}.

The game state is given below. Respond only with the next move in the format "original_position-destination_position" (eg: 12-21), where original_position corresponds to a piece that can be moved, and destination_position corresponds to the final position of the piece. Do not include any explanations or additional text.
"""
        prompt += f"\nBoard:\n{str(board)}\n\n"
        prompt += "Your move: "
        prompt += f"{move.from_coord_num()}-"
        prompts.append(prompt)
    return prompts

def get_piece_movement_dataset(num_games: int = 500, num_samples_per_game: int = 10):
    games: list[Xiangqi] = [_generate_game() for _ in range(num_games)]
    dataset: list[str] = []
    for game in games:
        dataset.extend(_get_prompt(game, num_samples_per_game))
    return Dataset.from_dict({"query": dataset})
