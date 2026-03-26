import matplotlib.pyplot as plt

def main():
    y_1 = [0, 310, 108, 227]
    y_2 = [35, 290, 52, 217]
    y_3 = [0, 357, 23, 252]
    y_4 = [0, 289, 1, 52]
    y_5 = [9, 338, 62, 76]
    y_6 = [0, 362, 44, 280]
    y_1 = [y / 5 for y in y_1]
    y_2 = [y / 5 for y in y_2]
    y_3 = [y / 5 for y in y_3]
    y_4 = [y / 5 for y in y_4]
    y_5 = [y / 5 for y in y_5]
    y_6 = [y / 5 for y in y_6]
    x = ["pre-training", "D", "S", "final"]
    title_1 = "google/gemma-2-2b-it"
    title_2 = "LiquidAI/LFM2-350M"
    title_3 = "meta-llama/Llama-3.1-8B-Instruct"
    title_4 = "openai/gpt-oss-20b"
    title_5 = "Qwen/Qwen2.5-1.5B-Instruct"
    title_6 = "Qwen/Qwen3-8B"
    overall_title = "(C3) for D + S"
    fig_name = "C-DS.png"

    y_values = [y_1, y_2, y_3, y_4, y_5, y_6]
    titles = [title_1, title_2, title_3, title_4, title_5, title_6]

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True)
    axes = axes.flatten()

    for ax, y, title in zip(axes, y_values, titles):
        ax.plot(x, y, marker="o", linewidth=2)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Stage")
        ax.set_ylabel("Score")
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(overall_title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(fig_name, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()