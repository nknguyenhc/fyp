from dataclasses import field, dataclass

@dataclass
class CLTrainingArguments:
    step: str = field(
        default="vls",
    )
