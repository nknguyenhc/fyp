import re
import torch

from ttt_dataset import TicTacToe

def get_question_and_answer(text: str):
    question, answer = text.split("0, 1, or 2 counting from left to right.", 1)
    return question, answer

def get_board(question: str) -> TicTacToe:
    regex = re.compile(r"((X|O|-) (X|O|-) (X|O|-)\n(X|O|-) (X|O|-) (X|O|-)\n(X|O|-) (X|O|-) (X|O|-))")
    match = regex.search(question)
    if match:
        board_str = match.group(0)
        return TicTacToe.from_string(board_str)
    raise ValueError(f"Invalid question format: {question}")

def get_score(question: str, answer: str) -> float:
    board = get_board(question)
    regex = re.compile(r"Final Answer\**:\** (\d), ?(\d)")
    # Find all matches and obtain the last match
    matches = regex.findall(answer)
    if not matches:
        return -30  # Very bad
    last_match = matches[-1]
    row, col = map(int, last_match)
    if board.is_valid_move((row, col)):
        return 10
    else:
        return -10

class ForwardResult:
    def __init__(self, item):
        self.hidden_states = [item]

class TTTReward:
    def __init__(self, tokenizer):
        self.base_model_prefix = 'huh'
        self.huh = self.forward
        self.tokenizer = tokenizer

    def forward(self, *args, **kwargs):
        decoded = self.tokenizer.batch_decode(kwargs['input_ids'], skip_special_tokens=True)
        questions_and_answers = [get_question_and_answer(text) for text in decoded]
        scores = torch.tensor([get_score(question, answer) for question, answer in questions_and_answers], dtype=torch.float32).unsqueeze(1)
        item = scores.repeat(1, kwargs['input_ids'].shape[1])
        return ForwardResult(item.to('cuda'))

    def score(self, scores: torch.Tensor):
        return scores

    def modules(self):
        return []
    
    def to(self, _):
        return self