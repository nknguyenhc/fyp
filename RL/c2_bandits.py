import random
import matplotlib.pyplot as plt

from ttt import Tictactoe, Agent, Orchestrator

class ActionValueMethod(Agent):
    def __init__(self, lr: float = 0.1, epsilon: float = 0.1):
        self.action_values = [[[0] * 9 for _ in range(512)] for _ in range(512)]
        self.steps: list[tuple[int, int, int, int]] = []
        self.lr = lr
        self.epsilon = epsilon

    def new_simulation(self):
        self.steps = []

    def get_action(self, board: Tictactoe, player: int, training: bool = True) -> int:
        best_actions = []
        best_value = float('-inf')
        x_value, o_value = board.get_numbers()
        if training and random.random() < self.epsilon:
            row, col = random.randint(0, 2), random.randint(0, 2)
            action_num = row * 3 + col
            self.steps.append((x_value, o_value, action_num, player))
            return (row, col)
        for row in range(3):
            for col in range(3):
                action_num = row * 3 + col
                action_value = self.action_values[x_value][o_value][action_num]
                if action_value > best_value:
                    best_value = action_value
                    best_actions = [(row, col)]
                elif action_value == best_value:
                    best_actions.append((row, col))
        best_action = random.choice(best_actions)
        if training:
            action_num = best_action[0] * 3 + best_action[1]
            self.steps.append((x_value, o_value, action_num, player))
        return best_action
    
    def update_invalid_move(self, reward: float):
        x_value, o_value, action_num, player = self.steps[-1]
        self.steps = self.steps[:-1]
        self.action_values[x_value][o_value][action_num] += self.lr * (reward - self.action_values[x_value][o_value][action_num])
    
    def update_action_value(self, reward: float):
        for x_value, o_value, action_num, player in self.steps:
            if player == 1:
                self.action_values[x_value][o_value][action_num] += self.lr * (reward - self.action_values[x_value][o_value][action_num])
            else:
                self.action_values[x_value][o_value][action_num] += self.lr * (-reward - self.action_values[x_value][o_value][action_num])

def main():
    epsilons = [0.01, 0.1, 0.2, 0.3, 0.4]
    evaluation_results = []
    breakpoints = 100
    for epsilon in epsilons:
        agent = ActionValueMethod(epsilon=epsilon)
        orchestrator = Orchestrator(agent)
        evaluation_results.append(orchestrator.run(breakpoints=breakpoints))
    plt.figure()
    for evaluation_result, epsilon in zip(evaluation_results, epsilons):
        plt.plot([i * breakpoints for i in range(len(evaluation_result))], evaluation_result, label=f"epsilon={epsilon}")
    plt.xlabel("Iteration")
    plt.ylabel("Evaluation Score")
    plt.title("Agent Evaluation")
    plt.legend()
    plt.savefig(f"c2_bandits.png")



if __name__ == "__main__":
   main()
