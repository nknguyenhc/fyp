from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import json
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

from ult_ttt import *

def extract_tool_call_from_response(response: str):
    """
    Extract the tool call from the model's response.
    JSON loads all possible substrings and returns the first valid one with "name": "something"
    """
    for start in range(len(response)):
        for end in range(start + 1, len(response) + 1):
            substring = response[start:end]
            try:
                data = json.loads(substring)
                if type(data) != dict:
                    continue
                if "name" in data and ("arguments" in data or "parameters" in data):
                    return data
            except json.JSONDecodeError:
                continue

class LLMModel:
    def __init__(self, model: str, trust_remote_code: bool):
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
        self.model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=trust_remote_code, device_map="auto", torch_dtype="auto")
    
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

    def get_move_from_history(self, history: tuple[list[ImmutableState], list[Action]]) -> Action | None:
        prompt = self._get_prompt_from_history(history)
        print(f"Prompt: {prompt}", flush=True)
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tools=[get_valid_moves],
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self.model.device)
        result = self.model.generate(inputs, max_new_tokens=1000)
        response = self.tokenizer.decode(result[0][inputs.shape[1]:], skip_special_tokens=True)
        tool_call = extract_tool_call_from_response(response)
        if tool_call is None:
            return self._parse_response_from_history(response)
        function_name = tool_call["name"]
        arguments = tool_call["parameters"] if "parameters" in tool_call else tool_call["arguments"]
        print(f"Function to call: {function_name} with arguments {arguments}", flush=True)
        if function_name != "get_valid_moves":
            raise ValueError(f"Unknown function name: {function_name}")
        state_str = arguments["state"]
        prev_move = int(arguments["prev_move"])
        valid_moves = get_valid_moves(state_str, prev_move)
        print(f"Valid moves from tool: {valid_moves}", flush=True)
        messages += [
            {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": function_name, "arguments": arguments}}]},
            {"role": "tool", "name": function_name, "content": str(valid_moves)}
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self.model.device)
        result = self.model.generate(inputs, max_new_tokens=1000)
        response = self.tokenizer.decode(result[0][inputs.shape[1]:], skip_special_tokens=True)
        print(f"Response: {response}", flush=True)
        return self._parse_response_from_history(response)
    
    def _parse_response_from_history(self, response: str) -> Action | None:
        for i in range(min(len(response), 10)):
            if i + 1 < len(response):
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

class Experiment:
    def __init__(self, model_name: str, trust_remote_code: bool):
        self.model = LLMModel(model_name, trust_remote_code)
        self.model_name = model_name

    def run(self, num_games: int = 500):
        tool_errors = 0
        invalid_format = 0
        invalid_moves = 0
        valid_moves = 0
        for _ in range(num_games):
            game_history: tuple[list[ImmutableState], list[Action]] = generate_game_history()
            try:
                move = self.model.get_move_from_history(game_history)
            except ValueError as e:
                print(f"Error getting move: {e}", flush=True)
                tool_errors += 1
                continue
            if move is None:
                print("Invalid format", flush=True)
                invalid_format += 1
                continue
            if not is_valid_action(game_history[0][-1], move):
                print("Invalid move", flush=True)
                invalid_moves += 1
                continue
            print("Valid move", flush=True)
            valid_moves += 1

        with open(f"result.{self.model_name.replace('/', '.')}.txt", "w") as f:
            f.write(f"Tool errors: {tool_errors}\n")
            f.write(f"Invalid format: {invalid_format}\n")
            f.write(f"Invalid moves: {invalid_moves}\n")
            f.write(f"Valid moves: {valid_moves}\n")

if __name__ == '__main__':
    model_name = sys.argv[1]
    trust_remote_code = sys.argv[2].lower() == 'true'
    experiment = Experiment(model_name, trust_remote_code)
    experiment.run()
