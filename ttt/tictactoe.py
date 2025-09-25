import random
from copy import deepcopy
from transformers import pipeline
import re
import sys
import torch

print(f"Available GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)} - {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")

class TicTacToe:
    def __init__(self, board: list[list[int]], turn: int = 1):
        self.board = board
        self.turn = turn

    def generate_board() -> "TicTacToe":
        num_moves = random.randint(0, 4)
        board = [[0] * 3 for _ in range(3)]
        ttt = TicTacToe(board)
        for i in range(num_moves):
            ttt = ttt.rand_move()
        return ttt
    
    def generate_game_history() -> tuple[list["TicTacToe"], list[tuple[int, int]]]:
        ttt = TicTacToe([[0] * 3 for _ in range(3)])
        history = [ttt]
        moves = []
        num_moves = random.randint(1, 4)
        for _ in range(num_moves):
            free_positions = [(i, j) for i in range(3) for j in range(3) if ttt.board[i][j] == 0]
            assert free_positions, "No more free positions"
            i, j = random.choice(free_positions)
            new_board = deepcopy(ttt.board)
            new_board[i][j] = ttt.turn
            ttt = TicTacToe(new_board, -ttt.turn)
            history.append(ttt)
            moves.append((i, j))
        return history, moves

    def rand_move(self):
        free_positions = [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == 0]
        if not free_positions:
            return self
        i, j = random.choice(free_positions)
        board = deepcopy(self.board)
        board[i][j] = self.turn
        return TicTacToe(board, -self.turn)

    def is_valid_move(self, move: tuple[int, int]) -> bool:
        x, y = move
        return 0 <= x < 3 and 0 <= y < 3 and self.board[x][y] == 0
    
    def _format_number(self, num: int) -> str:
        if num == 1:
            return "X"
        elif num == -1:
            return "O"
        return "-"

    def __repr__(self):
        return f"""
{self._format_number(self.board[0][0])} {self._format_number(self.board[0][1])} {self._format_number(self.board[0][2])}
{self._format_number(self.board[1][0])} {self._format_number(self.board[1][1])} {self._format_number(self.board[1][2])}
{self._format_number(self.board[2][0])} {self._format_number(self.board[2][1])} {self._format_number(self.board[2][2])}
"""
    
    def __str__(self):
        return self.__repr__()

    def get_turn(self) -> str:
        return self._format_number(self.turn)

class LLMModel:
    def __init__(self, model: str):
        self.pipe = pipeline("text-generation", model=model, torch_dtype="auto", model_kwargs={"device_map":"auto"})
        self.pipe.model.generation_config.pad_token_id = self.pipe.tokenizer.eos_token_id
    
    def _get_prompt(self, board: TicTacToe):
        return f"""
You are an expert at playing tictactoe. Your current board is:

{board}

You are playing as {board.get_turn()}.

Determine your best next move as {board.get_turn()}. Explain your answer within 300 words, and at the end, return your move in the format `Final Answer: row, column`

Where row is 0, 1, or 2 counting from top down, and column is 0, 1, or 2 counting from left to right.
"""
    
    def get_moves(self, boards: list[TicTacToe]) -> list[tuple[int, int] | None]:
        prompts = [[{"role": "user", "content": self._get_prompt(board)}] for board in boards]
        for i, prompt in enumerate(prompts):
            print(f"Prompt {i}: {prompt[0]['content']}", flush=True)
        results = self.pipe(prompts, max_new_tokens=2000, do_sample=True, temperature=0.3)
        responses = [result[0]['generated_text'][1]["content"] for result in results]
        for i, response in enumerate(responses):
            print(f"Response {i}: {response}", flush=True)
        return [self._parse_response(response) for response in responses]

    def _parse_response(self, response: str) -> tuple[int, int] | None:
        regex = re.compile(r"Final Answer\**:\** (\d), ?(\d)")
        match = regex.search(response)
        if match:
            row, col = map(int, match.groups())
            return row, col
        return None
    
    def _get_prompt_from_history(self, history: tuple[list[TicTacToe], list[tuple[int, int]]]) -> str:
        boards = history[0]
        moves = history[1]
        prompt = """You are an expert at playing tictactoe. The tictactoe board is a 3x3 grid where each cell can be empty (-), contain an X, or contain an O. To indicate a move,
* Number 1 for top-left
* Number 2 for top-middle
* Number 3 for top-right
* Number 4 for middle-left
* Number 5 for middle-middle
* Number 6 for middle-right
* Number 7 for bottom-left
* Number 8 for bottom-middle
* Number 9 for bottom-right

The game history is given below. Respond only with the next move by indicating the number corresponding to the cell where you want to place your mark. Do not include any explanations or additional text.

"""
        prompt += f"Board:\n{boards[0]}"
        for board, move in zip(boards[1:], moves):
            move_num = 3 * move[0] + move[1] + 1
            prompt += f"\nMove: {move_num}\nBoard:\n{board}"
        prompt += "\nMove: "
        return prompt
    
    def get_moves_from_histories(self, histories: list[tuple[list[TicTacToe], list[tuple[int, int]]]]) -> list[tuple[int, int] | None]:
        prompts = [self._get_prompt_from_history(history) for history in histories]
        for i, prompt in enumerate(prompts):
            print(f"Prompt {i}: {prompt}", flush=True)
        results = self.pipe(prompts, max_new_tokens=200, do_sample=True, temperature=0.3)
        responses = [result[0]['generated_text'][len(prompt):] for result, prompt in zip(results, prompts)]
        for i, response in enumerate(responses):
            print(f"Response {i}: {response}", flush=True)
        return [self._parse_response_from_history(response) for response in responses]
    
    def _parse_response_from_history(self, response: str) -> tuple[int, int] | None:
        for char in response:
            if char in '123456789':
                move_num = int(char) - 1
                row = move_num // 3
                col = move_num % 3
                return row, col
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
            game_histories: list[tuple[list[TicTacToe], list[tuple[int, int]]]] = [TicTacToe.generate_game_history() for _ in range(batch_size)]
            moves = self.model.get_moves_from_histories(game_histories)
            for game, move in zip(game_histories, moves):
                if not move:
                    invalid_format += 1
                    continue
                if not game[0][-1].is_valid_move(move):
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
