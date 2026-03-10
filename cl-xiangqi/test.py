from transformers import pipeline
import sys
import random

from xiangqi import Xiangqi, Move

class LLMModel:
    def __init__(self, model_name: str, trust_remote_code: bool):
        self.pipe = pipeline("text-generation", model=model_name, torch_dtype="auto", model_kwargs={"device_map":"auto"}, trust_remote_code=trust_remote_code)
        if self.pipe.model.config.pad_token_id is None:
            self.pipe.model.generation_config.pad_token_id = self.pipe.tokenizer.eos_token_id
    
    def _get_prompt_from_board(self, board: Xiangqi) -> str:
        prompt = f"""You are an expert at playing Chinese Chess. The board is 10x9 where each cell can be empty (--), contain a red piece (_0), or contain a black piece (_1). The board position numbers are as follows:

 1  2  3  4  5  6  7  8  9
10 11 12 13 14 15 16 17 18
19 20 21 22 23 24 25 26 27
28 29 30 31 32 33 34 35 36
37 38 39 40 41 42 43 44 45
46 47 48 49 50 51 52 53 54
55 56 57 58 59 60 61 62 63
64 65 66 67 68 69 70 71 72
73 74 75 76 77 78 79 80 81
82 83 84 85 86 87 88 89 90

The pieces are represented as follows:
- King: K0 (Red), K1 (Black)
- Advisor: A0 (Red), A1 (Black)
- Elephant: E0 (Red), E1 (Black)
- Horse: H0 (Red), H1 (Black)
- Rook: R0 (Red), R1 (Black)
- Cannon: C0 (Red), C1 (Black)
- Pawn: P0 (Red), P1 (Black)

The red pieces are initially at the bottom half of the board, while the black pieces are initially at the top half of the board. The goal of the game is to checkmate the opponent's king, while protecting your own king. You are currently playing as {"Red" if board.turn else "Black"}.

The game state is given below. Respond only with the next move in the format "original_position-destination_position" (eg: 12-21), where original_position corresponds to a piece that can be moved, and destination_position corresponds to the final position of the piece. Do not include any explanations or additional text.
"""
        prompt += f"\nBoard:\n{str(board)}\n\n"
        prompt += "Your move: "
        return prompt

    def _parse_response(self, xiangqi: Xiangqi, response: str) -> Move | None:
        for i in range(len(response), 1, -1):
            try:
                original_pos_str, dest_pos_str = response[:i].strip().split("-")
                original_pos = int(original_pos_str) - 1
                dest_pos = int(dest_pos_str) - 1
                if not (0 <= original_pos < 90 and 0 <= dest_pos < 90):
                    return None
                original_cell = (original_pos // 9, original_pos % 9)
                dest_cell = (dest_pos // 9, dest_pos % 9)
                return xiangqi.parse_move(original_cell, dest_cell)
            except ValueError:
                try:
                    original_pos = int(response[:i].strip()) - 1
                    if not (0 <= original_pos < 90):
                        return None
                    original_cell = (original_pos // 9, original_pos % 9)
                    return xiangqi.parse_move(original_cell, None)
                except ValueError:
                    continue
    
    def get_moves_from_boards(self, boards: list[Xiangqi]) -> list[Move | None]:
        prompts = [self._get_prompt_from_board(board) for board in boards]
        for i, prompt in enumerate(prompts):
            print(f"Prompt {i+1}:\n{prompt}", flush=True)
        results = self.pipe(prompts, max_new_tokens=5)
        responses = [result[0]['generated_text'][len(prompt):] for result, prompt in zip(results, prompts)]
        for i, response in enumerate(responses):
            print(f"Response {i+1}:\n{response}", flush=True)
        return [self._parse_response(board, response) for board, response in zip(boards, responses)]

class Experiment:
    def __init__(self, model_name: str, trust_remote_code: bool):
        self.model = LLMModel(model_name, trust_remote_code)
        self.model_name = model_name
    
    def run(self, num_games: int = 500, batch_size: int = 10):
        invalid_format = 0
        invalid_moves = 0
        enemy_pieces = 0
        friendly_pieces = 0
        starting_valid_positions = 0
        valid_moves = 0
        for _ in range(num_games // batch_size):
            games = [self._generate_games() for _ in range(batch_size)]
            moves = self.model.get_moves_from_boards([game for game in games])
            for game, move in zip(games, moves):
                allowed_moves = game.actions()
                if move is None:
                    invalid_format += 1
                elif move in allowed_moves:
                    valid_moves += 1
                elif any(move.from_coords == amove.from_coords for amove in allowed_moves):
                    starting_valid_positions += 1
                elif game.has_friendly_piece(move.from_coords):
                    friendly_pieces += 1
                elif game.has_enemy_piece(move.from_coords):
                    enemy_pieces += 1
                else:
                    invalid_moves += 1
        
        with open(f"result.{self.model_name.replace('./', '').replace('/', '.')}.txt", 'w') as f:
            f.write(f"Invalid format: {invalid_format}\n")
            f.write(f"Invalid moves: {invalid_moves}\n")
            f.write(f"Friendly pieces: {friendly_pieces}\n")
            f.write(f"Enemy pieces: {enemy_pieces}\n")
            f.write(f"Starting valid positions: {starting_valid_positions}\n")
            f.write(f"Valid moves: {valid_moves}\n")
    
    def _generate_games(self) -> Xiangqi:
        num_moves = random.randint(10, 30)
        board = Xiangqi()
        for _ in range(num_moves):
            possible_moves = board.actions()
            if not possible_moves:
                break
            move = random.choice(possible_moves)
            board = board.move(move)
        if len(board.actions()) == 0:
            return self._generate_games()
        return board

if __name__ == "__main__":
    model_name = sys.argv[1]
    trust_remote_code = sys.argv[2].lower() == 'true'
    experiment = Experiment(model_name, trust_remote_code)
    experiment.run()
