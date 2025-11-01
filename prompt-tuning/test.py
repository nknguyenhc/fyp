from transformers import AutoTokenizer, AutoModelForCausalLM
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from accelerate import PartialState
import torch
import sys

from ult_ttt import *
from ttt_dataset import get_init_prompt
from script import ModelWrapper

class LLMModel:
    def __init__(self, model: str, trust_remote_code: bool):
        self.tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

        model_kwargs = dict(
            torch_dtype=torch.float16,
            device_map=None,
            trust_remote_code=trust_remote_code,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model,
            **model_kwargs,
        )
        self.model = ModelWrapper(base_model, get_init_prompt(self.tokenizer))

        # Load the soft prompts
        self.model.soft_tokens.data = torch.load(f"{model.replace('/', '.')}.soft_prompt.pt")

        accel = PartialState()
        device = accel.device if hasattr(accel, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
    
    def _get_prompt_from_history(self, history: tuple[list[ImmutableState], list[Action]]) -> str:
        boards = history[0]
        moves = history[1]
        prompt = f"Board:\n{convert_board_to_string(boards[0].board)}"
        for board, move in zip(boards[1:], moves):
            move_num = 27 * move[0] + 9 * move[1] + 3 * move[2] + move[3] + 1
            prompt += f"\nMove: {move_num}\nBoard:\n{convert_board_to_string(board.board)}"
        prompt += "\nMove: "
        return prompt

    def get_moves_from_histories(self, histories: list[tuple[list[ImmutableState], list[Action]]]) -> list[Action | None]:
        prompts = [self._get_prompt_from_history(history) for history in histories]
        for i, prompt in enumerate(prompts):
            print(f"Prompt {i}: {prompt}", flush=True)
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        if "token_type_ids" in inputs:
            inputs.pop("token_type_ids")
        results = self.model.generate(**inputs, max_new_tokens=2, do_sample=True, temperature=0.3)
        context_length = inputs['input_ids'].shape[1]
        results = results[:, context_length:]
        responses = [self.tokenizer.decode(result, skip_special_tokens=True) for result in results]
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
        self.model = LLMModel(model_name, trust_remote_code=trust_remote_code)
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
    trust_remote_code = sys.argv[2].lower() == "true"
    experiment = Experiment(model_name, trust_remote_code)
    experiment.run()
