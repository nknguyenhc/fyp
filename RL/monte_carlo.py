import matplotlib.pyplot as plt

from ttt import Tictactoe, Agent, Orchestrator

class MonteCarloAgent(Agent):
    def __init__(self):
        self._values = [[[0] * 9 for _ in range(512)] for _ in range(512)]
        self._counts = [[[1] * 9 for _ in range(512)] for _ in range(512)]
        self._steps: list[tuple[int, int, int, int]] = []
    
    def new_simulation(self):
        self._steps = []
    
    def get_action(self, board: Tictactoe, player: int, training: bool = True) -> tuple[int, int]:
        x_value, o_value = board.get_numbers()
        best_action = None
        best_value = float('-inf')
        for row in range(3):
            for col in range(3):
                action_num = row * 3 + col
                action_value = self._values[x_value][o_value][action_num] / self._counts[x_value][o_value][action_num]
                if action_value > best_value:
                    best_value = action_value
                    best_action = (row, col)
        if training:
            action_num = best_action[0] * 3 + best_action[1]
            self._steps.append((x_value, o_value, action_num, player))
        return best_action
    
    def update_invalid_move(self, reward: float):
        x_value, o_value, action_num, _ = self._steps[-1]
        self._steps = self._steps[:-1]
        self._values[x_value][o_value][action_num] += reward
        self._counts[x_value][o_value][action_num] += 1
    
    def update_action_value(self, reward):
        for x_value, o_value, action_num, player in self._steps:
            if player == 1:
                self._values[x_value][o_value][action_num] += reward
            else:
                self._values[x_value][o_value][action_num] += -reward
            self._counts[x_value][o_value][action_num] += 1

def main():
    breakpoints = 100
    agent = MonteCarloAgent()
    orchestrator = Orchestrator(agent)
    evaluation_results = orchestrator.run(num_iterations=20000, breakpoints=breakpoints, random_starts=True)
    plt.figure()
    plt.plot([i * breakpoints for i in range(len(evaluation_results))], evaluation_results)
    plt.xlabel("Iteration")
    plt.ylabel("Evaluation Score")
    plt.title("Monte Carlo")
    plt.savefig("monte_carlo.png")
    orchestrator.test()


if __name__ == '__main__':
    main()
