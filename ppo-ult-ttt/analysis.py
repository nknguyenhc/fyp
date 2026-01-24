import ast
import matplotlib.pyplot as plt

def analyse_line(series, line):
    try:
        line = line.decode('utf-8').strip()
        data = ast.literal_eval(line)
    except (ValueError, SyntaxError, UnicodeDecodeError):
        return
    if not isinstance(data, dict):
        return
    series.append(data)

def analyse_file(filepath):
    series = []
    with open(filepath, 'rb') as f:
        text = f.readlines()
        for line in text:
            analyse_line(series, line)
    return series

def plot_graphs(series):
    keys = list(series[0].keys())
    episodes = [entry['episode'] for entry in series]
    for key in keys:
        values = [entry[key] for entry in series]
        plt.figure()
        plt.plot(episodes, values, label=key)
        plt.xlabel('Episode')
        plt.ylabel(key)
        plt.title(f'{key} over Episodes')
        plt.savefig(f'{key.replace("/", ".")}_over_episodes.png')
        plt.close()

def main():
    filepath = 'result.ppo.google.gemma-2-2b-it.out'
    series = analyse_file(filepath)
    plot_graphs(series)

if __name__ == "__main__":
    main()