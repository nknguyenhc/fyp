from transformers import pipeline
import sys

from ult_ttt import *

class LLMModel:
    def __init__(self, model: str):
        self.pipe = pipeline("text-generation", model=model, torch_dtype="auto", model_kwargs={"device_map":"auto"}, trust_remote_code=True)
        self.pipe.model.generation_config.pad_token_id = self.pipe.tokenizer.eos_token_id
    
    def _get_prompt_from_history(self, history: tuple[list[ImmutableState], list[Action]]) -> str:
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

    def get_moves_from_histories(self, histories: list[tuple[list[ImmutableState], list[Action]]]) -> list[Action | None]:
        prompts = [self._get_prompt_from_history(history) for history in histories]
        for i, prompt in enumerate(prompts):
            print(f"Prompt {i}: {prompt}", flush=True)
        results = self.pipe(prompts, max_new_tokens=200, do_sample=True, temperature=0.3)
        responses = [result[0]['generated_text'][len(prompt):] for result, prompt in zip(results, prompts)]
        for i, response in enumerate(responses):
            print(f"Response {i}: {response}", flush=True)
        return [self._parse_response_from_history(response) for response in responses]
    
    def _parse_response_from_history(self, response: str) -> Action | None:
        for i in range(min(len(response) - 1, 10)):
            move_str = response[i:i+2]
            if move_str.isdigit():
                move_num = int(move_str) - 1
                if not 0 <= move_num <= 80:
                    return None
                board = (move_num - 1) // 9
                cell = (move_num - 1) % 9
                return board // 3, board % 3, cell // 3, cell % 3
            char = response[i]
            if char.isdigit():
                move_num = int(char) - 1
                if not 0 <= move_num <= 80:
                    return None
                board = (move_num - 1) // 9
                cell = (move_num - 1) % 9
                return board // 3, board % 3, cell // 3, cell % 3
        return None

class Experiment:
    def __init__(self, model_name: str = "LiquidAI/LFM2-350M"):
        self.model = LLMModel(model_name)
        self.model_name = model_name

    def run(self, num_games: int = 500, batch_size: int = 10):
        invalid_format = 0
        invalid_moves = 0
        valid_moves = 0
        for _ in range(num_games // batch_size):
            game_histories: list[tuple[list[ImmutableState], list[Action]]] = [generate_game_history() for _ in range(batch_size)]
            moves = self.model.get_moves_from_histories(game_histories)
            for game, move in zip(game_histories, moves):
                if not move:
                    invalid_format += 1
                    continue
                if not is_valid_action(game[0][-1], move):
                    invalid_moves += 1
                    continue
                valid_moves += 1

        with open(f"result.{self.model_name.replace('/', '.')}.txt", "w") as f:
            f.write(f"Invalid format: {invalid_format}\n")
            f.write(f"Invalid moves: {invalid_moves}\n")
            f.write(f"Valid moves: {valid_moves}\n")

if __name__ == '__main__':
    model_name = sys.argv[1]
    experiment = Experiment(model_name)
    experiment.run()
