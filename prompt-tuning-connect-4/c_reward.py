import torch

from connect_4 import Board
from config import width

def get_board_and_answer(text: str):
    parts = text.split("Board:")
    question, answer = parts[-1].strip().split("Move:", 1)
    return question.strip(), answer.strip()

def get_score(question: str, answer: str) -> float:
    state: Board = Board.from_repr_string(question)
    try:
        num = int(answer) - 1
        if not 0 <= num < width:
            return -9
        if state.is_valid_action(num):
            return 3
        else:
            return -3
    except ValueError:
        return -9  # No readable response, very bad

class ForwardResult:
    def __init__(self, item):
        self.hidden_states = [item]

class CReward:
    def __init__(self, tokenizer):
        self.base_model_prefix = 'huh'
        self.huh = self.forward
        self.tokenizer = tokenizer
        self.device = next(iter(tokenizer.model.parameters())).device if hasattr(tokenizer, 'model') else 'cuda'

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            decoded = self.tokenizer.batch_decode(kwargs['input_ids'], skip_special_tokens=True)
            questions_and_answers = [get_board_and_answer(text) for text in decoded]
            scores = torch.tensor([get_score(question, answer) for question, answer in questions_and_answers],
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
