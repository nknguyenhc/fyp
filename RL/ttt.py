from tqdm import tqdm
import random
import math

def _get_winning_numbers() -> set[int]:
    winning_lines = [
        0b111000000,  # Row 1
        0b000111000,  # Row 2
        0b000000111,  # Row 3
        0b100100100,  # Column 1
        0b010010010,  # Column 2
        0b001001001,  # Column 3
        0b100010001,  # Diagonal \
        0b001010100,  # Diagonal /
    ]
    numbers = set()
    for i in range(512):
        for line in winning_lines:
            if (i & line) == line:
                numbers.add(i)
    return numbers

class Tictactoe:
    _winning_combinations = _get_winning_numbers()

    def __init__(self, x_value: int = None, o_value: int = None):
        if x_value is None or o_value is None:
            self._x_value = 0
            self._o_value = 0
        else:
            self._x_value = x_value
            self._o_value = o_value
    
    def _cell_str(self, cell: int) -> str:
        if cell == 1:
            return "X"
        elif cell == -1:
            return "O"
        else:
            return "."

    def __repr__(self) -> str:
        board = [
            [1 if (self._x_value & (1 << (i * 3 + j))) != 0 else -1 if (self._o_value & (1 << (i * 3 + j))) != 0 else 0 for j in range(3)]
            for i in range(3)
        ]
        return "\n".join(" ".join(self._cell_str(cell) for cell in row) for row in board)

    def __str__(self) -> str:
        return self.__repr__()
    
    def get_numbers(self) -> tuple[int, int]:
        return self._x_value, self._o_value
    
    def is_valid_move(self, row: int, col: int) -> bool:
        if 0 <= row < 3 and 0 <= col < 3:
            return (self._x_value & (1 << (row * 3 + col))) == 0 and (self._o_value & (1 << (row * 3 + col))) == 0
        return False
    
    def make_move(self, row: int, col: int, player: int) -> "Tictactoe":
        new_x_value = self._x_value
        new_o_value = self._o_value
        if player == 1:
            new_x_value |= (1 << (row * 3 + col))
        else:
            new_o_value |= (1 << (row * 3 + col))
        return Tictactoe(new_x_value, new_o_value)
    
    def check_winner(self, player: int) -> bool:
        if player == 1:
            return self._x_value in Tictactoe._winning_combinations
        else:
            return self._o_value in Tictactoe._winning_combinations

    def is_draw(self) -> bool:
        return (self._x_value | self._o_value) == 0b111111111
    
    def is_valid(self) -> bool:
        return (self._x_value & self._o_value) == 0

def _generate_board() -> Tictactoe:
    num_moves = random.randint(0, 4)
    player = 1
    board = Tictactoe()
    for _ in range(num_moves):
        actions: list[tuple[int, int]] = []
        for row in range(3):
            for col in range(3):
                if board.is_valid_move(row, col):
                    actions.append((row, col))
        action = random.choice(actions)
        board = board.make_move(*action, player)
        player = -player
    return board

class Agent:
    def __init__(self):
        raise NotImplementedError

    def new_simulation(self):
        raise NotImplementedError

    def get_action(self, board: Tictactoe, player: int) -> int:
        raise NotImplementedError
    
    def update_invalid_move(self, reward: float):
        raise NotImplementedError
    
    def update_action_value(self, reward: float):
        raise NotImplementedError

class RefAgent:
    def __init__(self):
        pass

    class Node:
        def __init__(self, state: Tictactoe, player: int, parent: "RefAgent.Node" = None, move: tuple[int, int] = None):
            self.state = state
            self.player = player
            self.parent = parent
            self.move = move
            self.children: list["RefAgent.Node"] = []
            self.N = 0
            self.U = 0
        
        def ucb(self) -> float:
            if self.N == 0:
                return float("inf")
            return -self.U / self.N + (2 * math.log(self.parent.N) / self.N) ** 0.5

        def select(self) -> "RefAgent.Node":
            if not self.children:
                return self
            return max(self.children, key=lambda child: child.ucb()).select()

        def expand(self) -> "RefAgent.Node":
            if self.state.check_winner(-self.player) or self.state.is_draw():
                return self
            for row in range(3):
                for col in range(3):
                    if not self.state.is_valid_move(row, col):
                        continue
                    new_state = self.state.make_move(row, col, self.player)
                    child = RefAgent.Node(new_state, -self.player, parent=self, move=(row, col))
                    self.children.append(child)
            return random.choice(self.children)
        
        def simulate(self) -> int:
            current_state = self.state
            current_player = self.player
            while not current_state.check_winner(-current_player) and not current_state.is_draw():
                valid_moves = []
                for row in range(3):
                    for col in range(3):
                        if current_state.is_valid_move(row, col):
                            valid_moves.append((row, col))
                assert len(valid_moves) > 0
                move = random.choice(valid_moves)
                current_state = current_state.make_move(*move, current_player)
                current_player = -current_player
            if current_state.is_draw():
                return 0
            elif current_player == self.player:
                return -1
            else:
                return 1
        
        def backpropagate(self, reward: int):
            self.N += 1
            self.U += reward
            if self.parent:
                self.parent.backpropagate(-reward)
    
    def get_action(self, board: Tictactoe, player: int, simulations: int = 200) -> tuple[int, int]:
        root = RefAgent.Node(board, player)
        for _ in range(simulations):
            leaf = root.select()
            child = leaf.expand()
            reward = child.simulate()
            child.backpropagate(reward)
        best_child = max(root.children, key=lambda child: child.N)
        return best_child.move

class Orchestrator:
    def __init__(self, agent: Agent):
        self.agent = agent

    def run(self, num_iterations: int = 10000, breakpoints: int = 100, random_starts: bool = False) -> None:
        evaluation_results = []
        for i in tqdm(range(num_iterations)):
            if i % breakpoints == 0:
                evaluation_results.append(self._evaluate())
            self.agent.new_simulation()
            board: Tictactoe = Tictactoe() if not random_starts else _generate_board()
            current_player = 1
            while True:
                action = self.agent.get_action(board, current_player, training=True)
                if not board.is_valid_move(*action):
                    self.agent.update_invalid_move(-10)
                    continue
                board = board.make_move(*action, current_player)
                if board.check_winner(current_player):
                    self.agent.update_action_value(current_player)
                    break
                if board.is_draw():
                    self.agent.update_action_value(0)
                    break
                current_player *= -1
        evaluation_results.append(self._evaluate())
        return evaluation_results
    
    def _evaluate(self, num_games: int = 50) -> int:
        total_reward = 0
        for i in range(num_games):
            ref_agent = RefAgent()
            board: Tictactoe = Tictactoe()
            current_player = 1
            ref_player = 1 if i % 2 == 0 else -1
            while True:
                if current_player == ref_player:
                    action = ref_agent.get_action(board, current_player)
                else:
                    action = self.agent.get_action(board, current_player, training=False)
                if not board.is_valid_move(*action):
                    assert current_player != ref_player
                    total_reward -= 10
                    break
                board = board.make_move(*action, current_player)
                if board.check_winner(current_player):
                    total_reward += 1 if current_player != ref_player else -1
                    break
                if board.is_draw():
                    total_reward += 0
                    break
                current_player *= -1
        return total_reward / num_games

    def test(self, num_games: int = 100):
        for i in range(num_games):
            self._test(1 if i % 2 == 0 else -1)

    def _test(self, human_turn: int):
        board = Tictactoe()
        current_player = 1
        while True:
            print(board)
            if current_player == human_turn:
                row, col = map(lambda x: int(x.strip()) - 1, input("Enter your move (row, col): ").split(","))
                if not board.is_valid_move(row, col):
                    print("Invalid move. Try again.")
                    continue
            else:
                action = self.agent.get_action(board, current_player, training=False)
                row, col = action
                print(f"Agent's move: {row + 1}, {col + 1}")
                if not board.is_valid_move(row, col):
                    print("Agent made an invalid move. Game over.")
                    break
            
            board = board.make_move(row, col, current_player)
            if board.check_winner(current_player):
                print(board)
                if current_player == human_turn:
                    print("You win!")
                else:
                    print("Agent wins!")
                break
            
            if board.is_draw():
                print(board)
                print("It's a draw!")
                break
            
            current_player *= -1
