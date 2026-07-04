import matplotlib.pyplot as plt

def main():
    # B-SD
    y_1 = [33, 211, 29, 172]
    y_2 = [61, 388, 132, 211]
    y_3 = [108, 361, 28, 395]
    y_4 = [4, 165, 25, 132]
    y_5 = [97, 257, 41, 98]
    y_6 = [60, 192, 64, 238]
    fig_name = "B-SD.png"

    # C-SD
    y_1 = [0, 104, 306, 207]
    y_2 = [35, 62, 289, 93]
    y_3 = [0, 37, 341, 20]
    y_4 = [0, 9, 286, 29]
    y_5 = [9, 66, 338, 51]
    y_6 = [0, 104, 353, 276]
    fig_name = "C-SD.png"

    # B-DS
    y_1 = [33, 18, 165, 148]
    y_2 = [61, 24, 382, 146]
    y_3 = [108, 36, 426, 136]
    y_4 = [4, 42, 190, 361]
    y_5 = [97, 130, 237, 189]
    y_6 = [60, 34, 219, 252]
    fig_name = "B-DS.png"

    # C-DS
    y_1 = [0, 310, 108, 227]
    y_2 = [35, 290, 52, 217]
    y_3 = [0, 357, 23, 252]
    y_4 = [0, 289, 1, 52]
    y_5 = [9, 338, 62, 76]
    y_6 = [0, 362, 44, 280]
    fig_name = "C-DS.png"

    y_1 = [y / 5 for y in y_1]
    y_2 = [y / 5 for y in y_2]
    y_3 = [y / 5 for y in y_3]
    y_4 = [y / 5 for y in y_4]
    y_5 = [y / 5 for y in y_5]
    y_6 = [y / 5 for y in y_6]
    x = ["pre-training", "valid start", "piece movement", "full"]
    title_1 = "google/gemma-2-2b-it"
    title_2 = "LiquidAI/LFM2-350M"
    title_3 = "meta-llama/Llama-3.1-8B-Instruct"
    title_4 = "openai/gpt-oss-20b"
    title_5 = "Qwen/Qwen2.5-1.5B-Instruct"
    title_6 = "Qwen/Qwen3-8B"

    y_values = [y_1, y_2, y_3, y_4, y_5, y_6]
    titles = [title_1, title_2, title_3, title_4, title_5, title_6]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot all model curves on one graph.
    for y, title in zip(y_values, titles):
        ax.plot(x, y, marker="o", linewidth=2, label=title)

    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
    ax.set_xlabel("Stage", fontsize=24)
    ax.set_ylabel("% moves with valid start", fontsize=24)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x, rotation=45, ha='right', rotation_mode="anchor")
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(fig_name, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()