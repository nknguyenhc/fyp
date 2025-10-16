import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

from script import PromptTuning
from ult_ttt import *

def _get_prompt_from_history(history: tuple[list[ImmutableState], list[Action]]) -> str:
    boards = history[0]
    moves = history[1]
    prompt = """You are an expert at playing ultimate tictactoe. The board is 9x9 consisting of 3x3 smaller subboards where each cell can be empty (-), contain an X, or contain an O. To indicate a move, Use the corresponding number for the cell where you want to place your mark:

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
    prompt += f"Board:\n{convert_board_to_string(boards[0].board)}"
    for board, move in zip(boards[1:], moves):
        move_num = 27 * move[0] + 9 * move[1] + 3 * move[2] + move[3] + 1
        prompt += f"\nMove: {move_num}\nBoard:\n{convert_board_to_string(board.board)}"
    prompt += "\nMove: "
    return prompt

def _parse_response_from_history(response: str) -> Action | None:
    for i in range(min(len(response) - 1, 10)):
        move_str = response[i:i+2]
        if move_str.isdigit():
            move_num = int(move_str) - 1
            if not 0 <= move_num <= 80:
                return None
            board = move_num // 9
            cell = move_num % 9
            return board // 3, board % 3, cell // 3, cell % 3
        char = response[i]
        if char.isdigit():
            move_num = int(char) - 1
            if not 0 <= move_num <= 80:
                return None
            board = move_num // 9
            cell = move_num % 9
            return board // 3, board % 3, cell // 3, cell % 3
    return None

def _get_moves_from_histories(tokenizer, model: PromptTuning, histories: list[tuple[list[ImmutableState], list[Action]]]) -> list[Action | None]:
    prompts = [_get_prompt_from_history(history) for history in histories]
    for i, prompt in enumerate(prompts):
        print(f"Prompt {i}: {prompt}", flush=True)
    encodings = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
    input_ids = encodings['input_ids'].to(model.soft_prompt.device)
    attention_mask = encodings['attention_mask'].to(model.soft_prompt.device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        generated_ids = torch.argmax(outputs.logits, dim=-1)
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    for i, response in enumerate(responses):
        print(f"Response {i}: {response}", flush=True)
    return [_parse_response_from_history(response) for response in responses]

def test_prompt_tuning(model_name: str, num_games: int = 500, batch_size: int = 10):
    model_path = model_name.replace('/', '_') + '_prompt_tuned.pth'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    model = PromptTuning(base_model, n_prompt_tokens=20)
    model.load_state_dict(torch.load(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    invalid_format = 0
    invalid_move = 0
    total_moves = 0
    for _ in range(num_games // batch_size):
        game_histories: list[tuple[list[ImmutableState], list[Action]]] = [generate_game_history() for _ in range(batch_size)]
        moves = _get_moves_from_histories(tokenizer, model, game_histories)
        for game, move in zip(game_histories, moves):
            if not move:
                invalid_format += 1
                continue
            if not is_valid_action(game[0][-1], move):
                invalid_move += 1
                continue
            total_moves += 1
        
        with open(f"result.{model_name.replace('/', '.')}.txt", "w") as f:
            f.write(f"Invalid format: {invalid_format}\n")
            f.write(f"Invalid moves: {invalid_move}\n")
            f.write(f"Valid moves: {total_moves}\n")


if __name__ == '__main__':
    model_name = sys.argv[1]
    test_prompt_tuning(model_name)
