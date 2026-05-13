import matplotlib.pyplot as plt

def main():
    y_1 = [33, 211, 29, 172]
    y_2 = [61, 388, 132, 211]
    y_3 = [108, 361, 28, 395]
    y_4 = [4, 165, 25, 132]
    y_5 = [97, 257, 41, 98]
    y_6 = [60, 192, 64, 238]
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
    fig_name = "B-SD.png"

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