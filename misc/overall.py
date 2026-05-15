import matplotlib.pyplot as plt


def main() -> None:
    labels = [
        "Avg % Valid Moves",
        "PPO + LoRA",
        "Curriculum 1",
        "Curriculum 2",
        "Curriculum 3",
        "Curriculum 4",
    ]
    values = [1.27, 43.97, 52.55, 55.97, 25.07, 32.47]

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#EECA3B"])

    ax.set_ylabel("Percentage", fontsize=14)
    ax.set_xticklabels(labels, fontsize=12, rotation=35, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0, 100)
    ax.set_title("Xiangqi", fontsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
