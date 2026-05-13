from dataclasses import dataclass, field

@dataclass
class GeneralArguments:
    game: str = field(
        default="ult-ttt",
    )
