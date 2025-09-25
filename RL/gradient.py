import random
import matplotlib.pyplot as plt
import math

from ttt import Tictactoe, Agent, Orchestrator

class GradientMethod(Agent):
    def __init__(self, lr: float = 0.1):
        self.H = [[[0] * 9 for _ in range(512)] for _ in range(512)]
        self.lr = lr
        self.R_total = 0
        self.k = 1
        self.steps: list[tuple[int, int, int, int]] = []
    
    def new_simulation(self):
        self.steps = []
    
    def get_action(self, board: Tictactoe, player: int, training: bool = True) -> int:
        r = random.random()
        x_value, o_value = board.get_numbers()
        denom = sum([math.e ** self.H[x_value][o_value][a] for a in range(9)])
        numerator = 0
        for row in range(3):
            for col in range(3):
                action_num = row * 3 + col
                numerator += math.e ** self.H[x_value][o_value][action_num]
                if r < numerator / denom:
                    if training:
                        self.steps.append((x_value, o_value, action_num, player))
                    return (row, col)
        # fallback
        if training:
            self.steps.append((x_value, o_value, action_num, player))
        return (row, col)

    def update_invalid_move(self, reward: float):
        x_value, o_value, action_num, player = self.steps[-1]
        self.steps = self.steps[:-1]
        self._update(x_value, o_value, action_num, reward)
    
    def _update(self, x_value: int, o_value: int, action_num: int, reward: float):
        reward = reward / 100
        denom = sum([math.e ** self.H[x_value][o_value][a] for a in range(9)])
        pi_a = math.e ** self.H[x_value][o_value][action_num] / denom
        self.H[x_value][o_value][action_num] += self.lr * reward * (1 - pi_a)
        for a in range(9):
            if a != action_num:
                pi_a = math.e ** self.H[x_value][o_value][a] / denom
                self.H[x_value][o_value][a] -= self.lr * reward * pi_a
        self.R_total += reward
        self.k += 1
    
    def update_action_value(self, reward: float):
        for x_value, o_value, action_num, player in self.steps:
            if player == 1:
                self._update(x_value, o_value, action_num, reward)
            else:
                self._update(o_value, x_value, action_num, -reward)

def main():
    lrs = [0.1, 0.2, 0.3, 0.4]
    evaluation_results = []
    breakpoints = 20
    for lr in lrs:
        agent = GradientMethod(lr=lr)
        orchestrator = Orchestrator(agent)
        evaluation_results.append(orchestrator.run(num_iterations=2000, breakpoints=breakpoints))
    plt.figure()
    for evaluation_result, lr in zip(evaluation_results, lrs):
        plt.plot([i * breakpoints for i in range(len(evaluation_result))], evaluation_result, label=f"lr={lr}")
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.title("Evaluation of Different Learning Rates")
    plt.legend()
    plt.savefig("gradient.png")


if __name__ == "__main__":
    main()
