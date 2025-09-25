import random
import matplotlib.pyplot as plt

from ttt import Tictactoe, Agent, Orchestrator

class DecreasingEpsilonAgent(Agent):
    def __init__(self, lr: float = 0.1):
        self.lr = lr
        self.k = 1
        self.steps: list[tuple[int, int, int, int]] = []
        self.action_values = [[[0] * 9 for _ in range(512)] for _ in range(512)]
    
    def new_simulation(self):
        self.steps = []
    
    def get_action(self, board: Tictactoe, player: int, training: bool = True) -> int:
        best_action = []
        best_value = float('-inf')
        x_value, o_value = board.get_numbers()
        if training and random.random() < 1 / self.k:
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
        # self.k += 1
    
    def update_action_value(self, reward: float):
        for x_value, o_value, action_num, player in self.steps:
            if player == 1:
                self.action_values[x_value][o_value][action_num] += self.lr * (reward - self.action_values[x_value][o_value][action_num])
            else:
                self.action_values[x_value][o_value][action_num] += self.lr * (-reward - self.action_values[x_value][o_value][action_num])
        self.k += 1

def main():
    agent = DecreasingEpsilonAgent()
    orchestrator = Orchestrator(agent)
    breakpoints = 100
    results = orchestrator.run(breakpoints=breakpoints)
    plt.plot([i * breakpoints for i in range(len(results))], results)
    plt.xlabel('Iteration')
    plt.ylabel('Evaluation Score')
    plt.title('Agent Evaluation')
    plt.savefig('decreasing_epsilon.png')
    orchestrator.test()


if __name__ == "__main__":
    main()
