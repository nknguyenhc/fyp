from dataclasses import dataclass, field

@dataclass
class GeneralArguments:
    game: str = field(
        default="ult-ttt",
    )

@dataclass
class CLTrainingArguments:
    step: str = field(
        default="vls",
    )
