import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from trl import PPOTrainer, ScriptArguments, PPOConfig, ModelConfig, get_quantization_config, get_kbit_device_map
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from accelerate import PartialState
import shutil
import gc
from peft import PromptTuningConfig

from ttt_dataset import get_dataset, init_prompt
from ttt_reward import TTTReward

def prepare_dataset(dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples["query"], padding=True, truncation=True)

    return dataset.map(tokenize_function, batched=True, remove_columns=['query'])

def train_prompt_tuning(
    n_prompt_tokens=100,
):
    """Train prompt tuning"""
    torch.cuda.empty_cache()
    gc.collect()

    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    _, training_args, model_args = parser.parse_args_into_dataclasses()
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    device_map = "auto" if quantization_config is None else get_kbit_device_map()
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    
    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    
    # Prepare dataset
    dataset = get_dataset()
    with PartialState().local_main_process_first():
        dataset = prepare_dataset(dataset, tokenizer)
    
    peft_config = PromptTuningConfig(
        task_type="CAUSAL_LM",
        num_virtual_tokens=n_prompt_tokens,
        token_dim=model.config.hidden_size,
        num_transformer_submodules=1,
        num_attention_heads=model.config.num_attention_heads,
        num_layers=model.config.num_hidden_layers,
        prompt_tuning_init="TEXT",
        prompt_tuning_init_text=init_prompt,
        tokenizer_name_or_path=model_args.model_name_or_path,
    )
    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=model,
        reward_model=TTTReward(tokenizer),
        value_model=TTTReward(tokenizer),
        ref_model=None,
        train_dataset=dataset,
        eval_dataset=dataset,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)

    trainer.generate_completions()

if __name__ == '__main__':
    # Train the model
    train_prompt_tuning()
