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

def _get_score(texts: tuple[str, str, str, str, str]) -> list[float]:
    prompt, _ = texts[4].split("Your move:", 1)
    _, board_str = prompt.rsplit("Board:\n", 1)
    board = Xiangqi.from_string(board_str.strip())
    turn_regex = r"You are currently playing as (Red|Black)."
    turn_match = re.search(turn_regex, prompt)
    if not turn_match:
        raise ValueError("Could not determine player's turn from the prompt:\n" + prompt)
    player_color = turn_match.group(1)
    board.turn = player_color == "Red"
    moves = board.actions()
    scores = []
    for i, text in enumerate(texts):
        parts = text.split("Your move:", 1)
        if len(parts) < 2:
            # This means that eos token was generated
            scores.append(0)
            continue
        _, response = parts
        if response == "":
            scores.append(0)
            continue
        response = response.lstrip(" ")
        for move in moves:
            if str(move) == response:
                scores.append(3)
                scores.extend([0] * (4 - i))
                return scores
            if str(move).startswith(response):
                scores.append(3)
                break
        else:
            for move in _all_moves:
                if move == response:
                    scores.append(-3)
                    scores.extend([0] * (4 - i))
                    return scores
                if move.startswith(response):
                    scores.append(-3)
                    break
            else:
                scores.extend([-9] * (5 - i))
                return scores
    return scores

class ForwardResult:
    def __init__(self, item):
        self.hidden_states = [item]

class FullReward:
    def __init__(self, tokenizer):
        self.base_model_prefix = 'huh'
        self.huh = self.forward
        self.tokenizer = tokenizer
        self.device = next(iter(tokenizer.model.parameters())).device if hasattr(tokenizer, 'model') else 'cuda'
    
    def forward(self, *args, **kwargs):
        with torch.no_grad():
            input_ids = kwargs['input_ids']
            batch_size, seq_len = input_ids.shape

            decoded = [self.tokenizer.batch_decode(input_ids[:, :seq_len - i], skip_special_tokens=True) for i in range(4, -1, -1)]
            scores = torch.tensor([_get_score(texts) for texts in zip(*decoded)],
                                  dtype=torch.bfloat16, device=self.device)
            item = torch.zeros((batch_size, seq_len), dtype=torch.bfloat16, device=self.device)
            for i in range(5):
                item[:, seq_len - 5 + i] = scores[:, i]
            return ForwardResult(item.unsqueeze(-1))
    
    def score(self, scores: torch.Tensor):
        return scores
    
    def modules(self):
        return []
    
    def to(self, device):
        self.device = device
        return self
