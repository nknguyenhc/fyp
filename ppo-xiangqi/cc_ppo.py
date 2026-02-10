from accelerate import PartialState
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from trl import PPOTrainer, ScriptArguments, PPOConfig, ModelConfig, get_peft_config
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
import shutil
import torch
import gc

from cc_dataset import get_dataset
from cc_reward import CCReward

def prepare_dataset(dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples["query"], padding=True, truncation=True)

    return dataset.map(tokenize_function, batched=True, remove_columns=['query'])

def main():
    torch.cuda.empty_cache()
    gc.collect()

    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    _, training_args, model_args = parser.parse_args_into_dataclasses()
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    dataset = get_dataset()
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
    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=model,
        reward_model=CCReward(tokenizer),
        value_model=CCReward(tokenizer),
        ref_model=None,
        train_dataset=dataset,
        eval_dataset=dataset,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)

    trainer.generate_completions()


if __name__ == "__main__":
    main()

