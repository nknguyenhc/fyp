from datasets import Dataset

from connect_4 import Board, generate_game_history_with_full_columns

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
