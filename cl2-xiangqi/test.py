from transformers import HfArgumentParser
from trl import ModelConfig

from piece_movement_test import Experiment as PMExperiment
from main_test import Experiment

from args import CLTrainingArguments

def main():
    parser = HfArgumentParser((ModelConfig, CLTrainingArguments))
    model_args, cl_args = parser.parse_args_into_dataclasses()
    match cl_args.step:
        case "vls":
            experiment = Experiment(model_args.model_name_or_path, model_args.trust_remote_code)
        case "pm":
            experiment = PMExperiment(model_args.model_name_or_path, model_args.trust_remote_code)
        case _:
            raise ValueError(f"Invalid step: {cl_args.step}")
    experiment.run(cl_args.step)

if __name__ == "__main__":
    main()
