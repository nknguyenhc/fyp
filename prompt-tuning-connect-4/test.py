from transformers import AutoTokenizer, AutoModelForCausalLM
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from accelerate import PartialState
import torch
import sys

from connect_4 import Board, generate_game_history_with_full_columns
from config import width
from pt_c import ModelWrapper
from c_dataset import get_init_prompt

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
        # self.model.soft_tokens.data = torch.load(f"{model.replace('/', '.')}.soft_prompt.pt")

        accel = PartialState()
        device = accel.device if hasattr(accel, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
    
    def _get_prompt_from_history(self, history: tuple[list[Board], list[int]]) -> str:
        boards = history[0]
        moves = history[1]
        prompt = f"Board:\n{str(boards[0])}"
        for board, move in zip(boards[1:], moves):
            prompt += f"\nMove: {move + 1}\nBoard:\n{str(board)}"
        prompt += "\nMove: "
        return prompt

    def get_moves_from_histories(self, histories: list[tuple[list[Board], list[int]]]) -> list[int | None]:
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
        self.model = LLMModel(model_name, trust_remote_code=trust_remote_code)
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
    trust_remote_code = sys.argv[2].lower() == "true"
    experiment = Experiment(model_name, trust_remote_code)
    experiment.run()
