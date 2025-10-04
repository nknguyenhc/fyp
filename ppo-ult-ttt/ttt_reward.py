import torch

from ult_ttt import *

def get_board_and_answer(text: str):
    parts = text.split("Board:")
    question, answer = parts[-1].strip().split("Move:", 1)
    _, prev_move = parts[-2].strip().split("Move:", 1)
    return prev_move.strip(), question.strip(), answer.strip()

def get_score(prev_move: str, question: str, answer: str) -> float:
    prev_num = int(prev_move) - 1
    assert 0 <= prev_num <= 80
    cell = prev_num % 9
    prev_local_action = (cell // 3, cell % 3)
    state = board_string_to_state(question, prev_local_action)
    try:
        num = int(answer) - 1
        if not 0 <= num <= 80:
            return -9
        board = num // 9
        cell = num % 9
        action = (board // 3, board % 3, cell // 3, cell % 3)
        if is_valid_action(state, action):
            return 3
        else:
            return -3
    except ValueError:
        return -9  # No readable response, very bad

class ForwardResult:
    def __init__(self, item):
        self.hidden_states = [item]

class TTTReward:
    def __init__(self, tokenizer):
        self.base_model_prefix = 'huh'
        self.huh = self.forward
        self.tokenizer = tokenizer
        self.device = next(iter(tokenizer.model.parameters())).device if hasattr(tokenizer, 'model') else 'cuda'

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            decoded = self.tokenizer.batch_decode(kwargs['input_ids'], skip_special_tokens=True)
            questions_and_answers = [get_board_and_answer(text) for text in decoded]
            scores = torch.tensor([get_score(prev_move, question, answer) for prev_move, question, answer in questions_and_answers],
                                  dtype=torch.float32).unsqueeze(1)
            item = torch.concat((torch.zeros((scores.shape[0], kwargs['input_ids'].shape[1] - 2)),
                                 scores.repeat(1, 2)), dim=1)
            return ForwardResult(item.to(self.device))

    def score(self, scores: torch.Tensor):
        return scores

    def modules(self):
        return []
    
    def to(self, device):
        self.device = device
        return self