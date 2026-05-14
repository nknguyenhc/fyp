from transformers import HfArgumentParser
from trl import ModelConfig

from scripts.ult_ttt_test import Experiment as UltTTTExperiment
from scripts.connect_4_test import Experiment as CExperiment
from scripts.xiangqi.main_test import Experiment as XQExperiment
from scripts.xiangqi.piece_movement_test import Experiment as XQPMExperiment
from scripts.args import GeneralArguments, CLTrainingArguments

def main():
    parser = HfArgumentParser((ModelConfig, GeneralArguments, CLTrainingArguments))
    model_args, general_args, cl_args = parser.parse_args_into_dataclasses()
    match general_args.game:
        case "ult-ttt":
            experiment = UltTTTExperiment(model_args.model_name_or_path, model_args.trust_remote_code)
        case "connect-4":
            experiment = CExperiment(model_args.model_name_or_path, model_args.trust_remote_code)
        case "xiangqi":
            match cl_args.step:
                case "vls":
                    experiment = XQExperiment(model_args.model_name_or_path, model_args.trust_remote_code)
                case "pm":
                    experiment = XQPMExperiment(model_args.model_name_or_path, model_args.trust_remote_code)
                case _:
                    raise ValueError(f"Invalid step: {cl_args.step}")
            experiment.run(cl_args.step)
            return
        case _:
            raise ValueError(f"Invalid game: {general_args.game}")
    experiment.run()

if __name__ == "__main__":
    main()
