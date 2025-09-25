import random

from ttt import Tictactoe, Agent, Orchestrator

def _get_player(x_value: int, o_value: int) -> int:
    if x_value.bit_count() > o_value.bit_count():
        return -1
    else:
        return 1

class PolicyIterationAgent(Agent):
    def __init__(self):
        self._policies = [[0 for _ in range(512)] for _ in range(512)]
        self._policy_values = [[0 for _ in range(512)] for _ in range(512)]

    def _policy_evaluate(self, gamma: float, delta: float):
        while True:
            cells = []
            diff = 0
            for x_value in range(512):
                for o_value in range(512):
                    old_value = self._policy_values[x_value][o_value]
                    current_board = Tictactoe(x_value, o_value)
                    current_player = _get_player(x_value, o_value)
                    action = self._policies[x_value][o_value]
                    if not current_board.is_valid():
                        self._policy_values[x_value][o_value] = 10
                        diff = max(diff, abs(10 - old_value))
                    elif current_board.check_winner(current_player):
                        self._policy_values[x_value][o_value] = 1
                        diff = max(diff, abs(1 - old_value))
                    elif current_board.check_winner(-current_player):
                        self._policy_values[x_value][o_value] = -1
                        diff = max(diff, abs(-1 - old_value))
                    elif current_board.is_draw():
                        self._policy_values[x_value][o_value] = 0
                        diff = max(diff, abs(0 - old_value))
                    elif not current_board.is_valid_move(action // 3, action % 3):
                        self._policy_values[x_value][o_value] = -10
                        diff = max(diff, abs(-10 - old_value))
                    else:
                        action = self._policies[x_value][o_value]
                        next_board = current_board.make_move(action // 3, action % 3, current_player)
                        next_x_value, next_o_value = next_board.get_numbers()
                        self._policy_values[x_value][o_value] = -gamma * self._policy_values[next_x_value][next_o_value]
                        diff = max(diff, abs(self._policy_values[x_value][o_value] - old_value))
                    if abs(self._policy_values[x_value][o_value] - old_value) != 0:
                        cells.append((x_value, o_value))
            if diff < delta:
                break
    
    def _policy_improvement_cell(self, x_value: int, o_value: int, delta: float):
        current_board = Tictactoe(x_value, o_value)
        current_player = _get_player(x_value, o_value)
        if not current_board.is_valid() \
            or current_board.check_winner(current_player) \
            or current_board.check_winner(-current_player) \
            or current_board.is_draw():
            return True
        action = self._policies[x_value][o_value]
        next_board = current_board.make_move(action // 3, action % 3, current_player)
        next_x_value, next_o_value = next_board.get_numbers()
        value = -self._policy_values[next_x_value][next_o_value]
        best_value = -self._policy_values[next_x_value][next_o_value]
        is_policy_stable = True
        for action in range(9):
            next_board = current_board.make_move(action // 3, action % 3, current_player)
            next_x_value, next_o_value = next_board.get_numbers()
            q = -self._policy_values[next_x_value][next_o_value]
            if not current_board.is_valid_move(action // 3, action % 3):
                q = -10
            if q > best_value:
                best_value = q
                self._policies[x_value][o_value] = action
                is_policy_stable = is_policy_stable and abs(q - value) < delta
        return is_policy_stable
    
    def _policy_improvement(self, delta: float):
        policy_stable = True
        count = 0
        for x_value in range(512):
            for o_value in range(512):
                is_policy_stable = self._policy_improvement_cell(x_value, o_value, delta)
                if not is_policy_stable:
                    count += 1
                policy_stable = policy_stable and is_policy_stable
        print(f"Improved {count} policies")
        return policy_stable
    
    def train(self, gamma: float = 1, delta: float = 0.001):
        i = 0
        policy_stable = False
        while not policy_stable:
            i += 1
            print(f"Iteration: {i}")
            self._policy_evaluate(gamma, delta)
            policy_stable = self._policy_improvement(delta)
    
    def get_action(self, board: Tictactoe, player: int, training: bool = False):
        x_value, o_value = board.get_numbers()
        action = self._policies[x_value][o_value]
        for row in range(3):
            for col in range(3):
                next_board = board.make_move(row, col, player)
                next_x, next_o = next_board.get_numbers()
                print(f"Value for ({row}, {col}): {self._policy_values[next_x][next_o]}")
        return action // 3, action % 3

def main():
    agent = PolicyIterationAgent()
    agent.train()
    orchestrator = Orchestrator(agent)
    orchestrator.test()


if __name__ == '__main__':
    main()
