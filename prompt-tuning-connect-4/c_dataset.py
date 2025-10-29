from datasets import Dataset
import torch

from connect_4 import Board, generate_game_history_with_full_columns

init_prompt = """You are an expert at playing connect-4. The board is 7 columns wide and 6 rows high, where each cell can be empty (-), contain an X, or contain an O. To indicate a move, respond with the number of the column (1 to 7) where you want to place your mark.

The game history is given below. Respond only with the next move by indicating the number corresponding to the column where you want to place your mark. Do not include any explanations or additional text.

"""

def get_init_prompt(tokenizer) -> torch.Tensor:
    inputs = tokenizer(init_prompt, return_tensors="pt")
    return inputs["input_ids"][0]

def get_dataset(num_samples: int = 1000):
    def get_prompt(history: tuple[list[Board], list[int]]) -> str:
        boards, moves = history
        prompt = f"Board:\n{str(boards[0])}"
        for board, move in zip(boards[1:], moves):
            prompt += f"\nMove: {move + 1}\nBoard:\n{str(board)}"
        prompt += "\nMove: "
        return prompt
    game_histories: list[tuple[list[Board], list[int]]] = [generate_game_history_with_full_columns() for _ in range(num_samples)]
    dataset: list[str] = [get_prompt(game_history) for game_history in game_histories]
    return Dataset.from_dict({"query": dataset})
