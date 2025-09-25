import matplotlib.pyplot as plt
import math

from ttt import Tictactoe, Agent, Orchestrator

class UcbMethod(Agent):
    def __init__(self, lr: float = 0.1):
        self.action_values = [[[0] * 9 for _ in range(512)] for _ in range(512)]
        self.steps: list[tuple[int, int, int, int]] = []
        self.lr = lr
        self.counts = [[[0] * 9 for _ in range(512)] for _ in range(512)]
    
    def new_simulation(self):
        self.steps = []
    
    def get_action(self, board: Tictactoe, player: int, training: bool = True) -> int:
        best_action = None
        best_value = float('-inf')
        x_value, o_value = board.get_numbers()
        total_count = sum(self.counts[x_value][o_value]) + 1
        for row in range(3):
            for col in range(3):
                action_num = row * 3 + col
                action_value = self.action_values[x_value][o_value][action_num]
                action_count = self.counts[x_value][o_value][action_num]
                value = action_value if not training \
                    else action_value + (2 * math.log(total_count) / action_count) ** 0.5 \
                    if action_count > 0 else float('inf')
                if value > best_value:
                    best_value = value
                    best_action = (row, col)
        if training:
            action_num = best_action[0] * 3 + best_action[1]
            self.steps.append((x_value, o_value, action_num, player))
        return best_action
    
    def update_invalid_move(self, reward: float):
        x_value, o_value, action_num, player = self.steps[-1]
        self.steps = self.steps[:-1]
        self.action_values[x_value][o_value][action_num] += self.lr * (reward - self.action_values[x_value][o_value][action_num])
        self.counts[x_value][o_value][action_num] += 1
    
    def update_action_value(self, reward: float):
        for x_value, o_value, action_num, player in self.steps:
            if player == 1:
                self.action_values[x_value][o_value][action_num] += self.lr * (reward - self.action_values[x_value][o_value][action_num])
            else:
                self.action_values[x_value][o_value][action_num] += self.lr * (-reward - self.action_values[x_value][o_value][action_num])
            self.counts[x_value][o_value][action_num] += 1

def main():
    agent = UcbMethod()
    orchestrator = Orchestrator(agent)
    breakpoints = 100
    evaluation_results = orchestrator.run(breakpoints=breakpoints)
    plt.figure()
    plt.plot([i * breakpoints for i in range(len(evaluation_results))], evaluation_results)
    plt.xlabel("Iteration")
    plt.ylabel("Evaluation Score")
    plt.title("UCB Agent Evaluation")
    plt.savefig("ucb.png")


if __name__ == "__main__":
    main()
