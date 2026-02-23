from dataclasses import field, dataclass

@dataclass
class ValidStartTrainingArguments:
    response_length_1: int = field(
        default=2,
    )

    response_length_2: int = field(
        default=5,
    )

    learning_rate_1: float = field(
        default=5e-5,
    )

    learning_rate_2: float = field(
        default=5e-5,
    )

    num_ppo_epochs_1: int = field(
        default=4,
    )

    num_ppo_epochs_2: int = field(
        default=4,
    )

    total_episodes_1: int = field(
        default=10000,
    )

    total_episodes_2: int = field(
        default=10000,
    )
