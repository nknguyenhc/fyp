from accelerate import PartialState
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from trl import PPOTrainer, ScriptArguments, PPOConfig, ModelConfig, get_peft_config
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
import shutil
import torch
import gc

from args import CLTrainingArguments
from valid_start_dataset import get_valid_start_dataset
from valid_start_reward import ValidPositionReward
from piece_movement_dataset import get_piece_movement_dataset
from full_dataset import get_full_dataset
from full_reward import FullReward
from piece_movement_test import Experiment as PMExperiment
from main_test import Experiment

def prepare_dataset(dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples["query"], padding=True, truncation=True)

    return dataset.map(tokenize_function, batched=True, remove_columns=['query'])

def main():
    torch.cuda.empty_cache()
    gc.collect()

    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig, CLTrainingArguments))
    _, training_args, model_args, cl_args = parser.parse_args_into_dataclasses()
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    
    model_kwargs = dict(
        device_map="auto",
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    match cl_args.step:
        case "vls":
            dataset = get_valid_start_dataset()
            reward_model = ValidPositionReward(tokenizer)
            value_model = ValidPositionReward(tokenizer)
        case "pm":
            dataset = get_piece_movement_dataset()
            reward_model = FullReward(tokenizer)
            value_model = FullReward(tokenizer)
        case "final":
            dataset = get_full_dataset()
            reward_model = FullReward(tokenizer)
            value_model = FullReward(tokenizer)
        case _:
            raise ValueError(f"Invalid step: {cl_args.step}")
    
    with PartialState().local_main_process_first():
        dataset = prepare_dataset(dataset, tokenizer)
    
    peft_config = get_peft_config(model_args)

    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=model,
        reward_model=reward_model,
        value_model=value_model,
        ref_model=None,
        train_dataset=dataset,
        eval_dataset=dataset,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)

    # Step 2: run intermediate test script
    experiment = PMExperiment(training_args.output_dir, model_args.trust_remote_code)
    experiment.run("piece_movement")
    experiment = Experiment(training_args.output_dir, model_args.trust_remote_code)
    experiment.run("valid_start")


if __name__ == "__main__":
    main()
