from transformers import HfArgumentParser
from trl import ModelConfig

from scripts.ult_ttt_test import Experiment as UltTTTExperiment
from scripts.pt_ult_ttt_test import Experiment as PTUltTTTExperiment
from scripts.connect_4_test import Experiment as CExperiment
from scripts.pt_connect_4_test import Experiment as PTCExperiment
from scripts.xiangqi.main_test import Experiment as XQExperiment
from scripts.xiangqi.piece_movement_test import Experiment as XQPMExperiment
from scripts.args import GeneralArguments, CLTrainingArguments

def main():
    parser = HfArgumentParser((ModelConfig, GeneralArguments, CLTrainingArguments))
    model_args, general_args, cl_args = parser.parse_args_into_dataclasses()
    match general_args.game:
        case "ult-ttt":
            match general_args.mode:
                case "ppo":
                    experiment = UltTTTExperiment(model_args.model_name_or_path, model_args.trust_remote_code)
                case "prompt-tuning":
                    experiment = PTUltTTTExperiment(model_args.model_name_or_path, model_args.trust_remote_code)
                case _:
                    raise ValueError(f"Invalid mode: {general_args.mode}")
        case "connect-4":
            match general_args.mode:
                case "ppo":
                    experiment = CExperiment(model_args.model_name_or_path, model_args.trust_remote_code)
                case "prompt-tuning":
                    experiment = PTCExperiment(model_args.model_name_or_path, model_args.trust_remote_code)
                case _:
                    raise ValueError(f"Invalid mode: {general_args.mode}")
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
