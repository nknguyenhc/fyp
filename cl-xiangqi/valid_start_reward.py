import re
import torch

from xiangqi import Xiangqi

def _find_all_moves() -> list[str]:
    all_moves = []
    for i in range(90):
        for j in range(90):
            all_moves.append(f"{i}-{j}")
    return all_moves

_all_moves = _find_all_moves()

def _get_score(text: str) -> float:
    prompt, response = text.split("Your move:", 1)
    _, board_str = prompt.rsplit("Board:\n", 1)
    board = Xiangqi.from_string(board_str.strip())
    turn_regex = r"You are currently playing as (Red|Black)."
    turn_match = re.search(turn_regex, prompt)
    if not turn_match:
        raise ValueError("Could not determine player's turn from the prompt:\n" + prompt)
    player_color = turn_match.group(1)
    board.turn = player_color == "Red"
    moves = board.actions()
    response = response.strip(" ")
    try:
        num = int(response) - 1
        if not 0 <= num <= 89:
            return -9
        from_row, from_col = divmod(num, 9)
        if any(move.from_coords == (from_row, from_col) for move in moves):
            return 3
        else:
            return -3
    except ValueError:
        return -9

class ForwardResult:
    def __init__(self, item):
        self.hidden_states = [item]

class ValidPositionReward:
    def __init__(self, tokenizer):
        self.base_model_prefix = 'huh'
        self.huh = self.forward
        self.tokenizer = tokenizer
        self.device = next(iter(tokenizer.model.parameters())).device if hasattr(tokenizer, 'model') else 'cuda'
    
    def forward(self, *args, **kwargs):
        with torch.no_grad():
            # Assuming 2 tokens are generated, and padding is on the left
            decoded = self.tokenizer.batch_decode(kwargs['input_ids'], skip_special_tokens=True)
            scores = torch.tensor([_get_score(text) for text in decoded],
                                  dtype=torch.bfloat16).unsqueeze(1)
            item = torch.concat((torch.zeros((scores.shape[0], kwargs['input_ids'].shape[1] - 2), dtype=torch.bfloat16),
                                 scores.repeat(1, 2)), dim=1)
            return ForwardResult(item.to(self.device))
    
    def score(self, scores: torch.Tensor):
        return scores
    
    def modules(self):
        return []
    
    def to(self, device):
        self.device = device
        return self
