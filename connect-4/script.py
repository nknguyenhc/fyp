from transformers import pipeline
import sys

from connect_4 import Board, generate_game_history_with_full_columns
from config import width

class LLMModel:
    def __init__(self, model: str, trust_remote_code: bool):
        self.pipe = pipeline(
            "text-generation",
            model=model,
            torch_dtype="auto",
            model_kwargs={"device_map":"auto"},
            trust_remote_code=trust_remote_code)
        self.pipe.model.generation_config.pad_token_id = self.pipe.tokenizer.eos_token_id
    
    def _get_prompt_from_history(self, history: tuple[list[Board], list[int]]) -> str:
        boards = history[0]
        moves = history[1]
        prompt = """You are an expert at playing connect-4. The board is 7 columns wide and 6 rows high, where each cell can be empty (-), contain an X, or contain an O. To indicate a move, respond with the number of the column (1 to 7) where you want to place your mark.

When a player places their mark in a column, it occupies the lowest available cell in that column. The objective of the game is to connect four of your marks in a row, either horizontally, vertically, or diagonally. You are not allowed to place a mark in a column that is already full.

The game history is given below. Respond only with the next move by indicating the number corresponding to the column where you want to place your mark. Do not include any explanations or additional text.

"""
        prompt += f"Board:\n{str(boards[0])}"
        for board, move in zip(boards[1:], moves):
            prompt += f"\nMove: {move + 1}\nBoard:\n{str(board)}"
        prompt += "\nMove: "
        return prompt

    def get_moves_from_histories(self, histories: list[tuple[list[Board], list[int]]]) -> list[int | None]:
        prompts = [self._get_prompt_from_history(history) for history in histories]
        for i, prompt in enumerate(prompts):
            print(f"Prompt {i}: {prompt}", flush=True)
        results = self.pipe(prompts, max_new_tokens=1, do_sample=True, temperature=0.3)
        responses = [result[0]['generated_text'][len(prompt):] for result, prompt in zip(results, prompts)]
        for i, response in enumerate(responses):
            print(f"Response {i}: {response}", flush=True)
        return [self._parse_response_from_history(response) for response in responses]
    
    def _parse_response_from_history(self, response: str) -> int | None:
        for i in range(min(len(response), 10)):
            char = response[i]
            if char.isdigit():
                move_num = int(char) - 1
                if not 0 <= move_num < width:
                    return None
                return move_num
        return None

class Experiment:
    def __init__(self, model_name: str, trust_remote_code: bool):
        self.model = LLMModel(model_name, trust_remote_code)
        self.model_name = model_name

    def run(self, num_games: int = 500, batch_size: int = 10):
        invalid_format = 0
        invalid_moves = 0
        valid_moves = 0
        for _ in range(num_games // batch_size):
            game_histories: list[tuple[list[Board], list[int]]] = [generate_game_history_with_full_columns() for _ in range(batch_size)]
            moves = self.model.get_moves_from_histories(game_histories)
            for game, move in zip(game_histories, moves):
                if move is None:
                    invalid_format += 1
                    continue
                if not game[0][-1].is_valid_action(move):
                    invalid_moves += 1
                    continue
                valid_moves += 1

        with open(f"result.{self.model_name.replace('/', '.')}.txt", "w") as f:
            f.write(f"Invalid format: {invalid_format}\n")
            f.write(f"Invalid moves: {invalid_moves}\n")
            f.write(f"Valid moves: {valid_moves}\n")

if __name__ == '__main__':
    model_name = sys.argv[1]
    trust_remote_code = sys.argv[2].lower() == 'true'
    experiment = Experiment(model_name, trust_remote_code)
    experiment.run()
