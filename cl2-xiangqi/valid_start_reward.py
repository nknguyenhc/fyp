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
    def __init__(self, tokenizer, num_token_generated: int):
        self.base_model_prefix = 'huh'
        self.huh = self.forward
        self.tokenizer = tokenizer
        self.device = next(iter(tokenizer.model.parameters())).device if hasattr(tokenizer, 'model') else 'cuda'
        self.num_token_generated = num_token_generated
    
    def forward(self, *args, **kwargs):
        with torch.no_grad():
            # Get the actual input length from the input_ids
            input_ids = kwargs['input_ids']
            batch_size, seq_len = input_ids.shape
            
            # Decode and get scores
            decoded = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            scores = torch.tensor([_get_score(text) for text in decoded],
                                  dtype=torch.bfloat16, device=self.device)
            
            # Create output tensor with shape [batch_size, seq_len]
            # Initialize all positions to 0
            item = torch.zeros((batch_size, seq_len), dtype=torch.bfloat16, device=self.device)
            
            # Place the score in the last position of each sequence
            # This represents the value/reward for completing the sequence
            for i in range(self.num_token_generated):
                item[:, -i-1] = scores
            
            return ForwardResult(item.unsqueeze(-1))
    
    def score(self, scores: torch.Tensor):
        return scores
    
    def modules(self):
        return []
    
    def to(self, device):
        self.device = device
        return self
