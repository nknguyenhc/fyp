from accelerate import PartialState
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from trl import PPOTrainer, ScriptArguments, PPOConfig, ModelConfig, get_peft_config
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
import shutil
import torch
import gc

from multi_step_args import MultiStepTrainingArguments
from piece_movement_dataset import get_piece_movement_dataset
from full_dataset import get_full_dataset
from full_reward import FullReward
from test import Experiment

# Flags
FIRST_STEP_ONLY = False

def prepare_dataset(dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples["query"], padding=True, truncation=True)

    return dataset.map(tokenize_function, batched=True, remove_columns=['query'])

def main():
    torch.cuda.empty_cache()
    gc.collect()

    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig, MultiStepTrainingArguments))
    _, training_args, model_args, multi_step_args = parser.parse_args_into_dataclasses()
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    # Step 1: train on valid starts
    dataset = get_piece_movement_dataset()
    model_kwargs = dict(
        device_map="auto",
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    with PartialState().local_main_process_first():
        dataset = prepare_dataset(dataset, tokenizer)
    peft_config = get_peft_config(model_args)

    # Assign training args
    training_args.response_length = multi_step_args.response_length_1
    training_args.learning_rate = multi_step_args.learning_rate_1
    training_args.num_ppo_epochs = multi_step_args.num_ppo_epochs_1
    training_args.total_episodes = multi_step_args.total_episodes_1

    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=model,
        reward_model=FullReward(tokenizer),
        value_model=FullReward(tokenizer),
        ref_model=None,
        train_dataset=dataset,
        eval_dataset=dataset,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)

    if FIRST_STEP_ONLY:
        return

    # Step 2: run intermediate test script
    model_args.model_name_or_path = training_args.output_dir
    experiment = Experiment(model_args.model_name_or_path, model_args.trust_remote_code)
    experiment.run()

    # Step 3: train on valid starts + valid moves
    dataset = get_full_dataset()
    model_kwargs = dict(
        device_map="auto",
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    with PartialState().local_main_process_first():
        dataset = prepare_dataset(dataset, tokenizer)

    peft_config = get_peft_config(model_args)

    # Assign training args
    training_args.response_length = multi_step_args.response_length_2
    training_args.learning_rate = multi_step_args.learning_rate_2
    training_args.num_ppo_epochs = multi_step_args.num_ppo_epochs_2
    training_args.total_episodes = multi_step_args.total_episodes_2

    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=model,
        reward_model=FullReward(tokenizer),
        value_model=FullReward(tokenizer),
        ref_model=None,
        train_dataset=dataset,
        eval_dataset=dataset,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
