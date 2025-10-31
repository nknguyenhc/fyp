import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from trl import PPOTrainer, ScriptArguments, PPOConfig, ModelConfig, get_quantization_config
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from accelerate import PartialState
import gc

from c_dataset import get_dataset, get_init_prompt
from c_reward import CReward

def prepare_dataset(dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples["query"], padding=True, truncation=True)
    return dataset.map(tokenize_function, batched=True, remove_columns=['query'])

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, init_prompt_input_ids: torch.Tensor):
        super().__init__()
        self.__model = model
        init_input_embeds = self.__model.get_input_embeddings()(init_prompt_input_ids)
        self.soft_tokens = torch.nn.Parameter(init_input_embeds)
        # Freeze the base model parameters
        for param in self.__model.parameters():
            param.requires_grad = False

    def forward(self, *args, **kwargs):
        if "input_ids" in kwargs:
            input_ids = kwargs["input_ids"]
            batch_size = input_ids.size(0)
            inputs_embeds = self.__model.get_input_embeddings()(input_ids)
            # Ensure soft tokens are on the same device as inputs_embeds
            soft_tokens = self.soft_tokens.to(inputs_embeds.device).expand(batch_size, -1, -1)
            inputs_embeds = torch.cat([soft_tokens, inputs_embeds], dim=1)
            kwargs["inputs_embeds"] = inputs_embeds
            del kwargs["input_ids"]
        elif "inputs_embeds" in kwargs:
            inputs_embeds = kwargs["inputs_embeds"]
            batch_size = inputs_embeds.size(0)
            # Ensure soft tokens are on the same device as inputs_embeds
            soft_tokens = self.soft_tokens.to(inputs_embeds.device).expand(batch_size, -1, -1)
            inputs_embeds = torch.cat([soft_tokens, inputs_embeds], dim=1)
            kwargs["inputs_embeds"] = inputs_embeds
        # Extend attention mask if provided
        if "attention_mask" in kwargs:
            attention_mask = kwargs["attention_mask"]
            soft_attention_mask = torch.ones(
                attention_mask.size(0),
                self.soft_tokens.size(0),
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([soft_attention_mask, attention_mask], dim=1)
            kwargs["attention_mask"] = attention_mask
        # Extend position ids if provided
        if "position_ids" in kwargs:
            position_ids = kwargs["position_ids"]
            soft_position_ids = torch.arange(
                self.soft_tokens.size(0), device=position_ids.device, dtype=position_ids.dtype
            ).unsqueeze(0).expand(position_ids.size(0), -1)
            position_ids = torch.cat([soft_position_ids, position_ids], dim=1)
            kwargs["position_ids"] = position_ids
        result = self.__model(*args, **kwargs)
        result.logits = result.logits[:, self.soft_tokens.size(0):, :]
        return result

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.__model, name)

def train_prompt_tuning():
    """Train prompt tuning"""
    torch.cuda.empty_cache()
    gc.collect()

    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    _, training_args, model_args = parser.parse_args_into_dataclasses()

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    # Do not shard the model; let Accelerate/DeepSpeed handle parallelism
    device_map = None

    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=model_args.trust_remote_code,
    )
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    
    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    base_model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    model = ModelWrapper(base_model, get_init_prompt(tokenizer))

    # Place the wrapper on the current process device
    accel = PartialState()
    device = accel.device if hasattr(accel, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Prepare dataset
    dataset = get_dataset()
    with PartialState().local_main_process_first():
        dataset = prepare_dataset(dataset, tokenizer)
    
    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=model,
        reward_model=CReward(tokenizer),
        value_model=CReward(tokenizer),
        ref_model=base_model,
        train_dataset=dataset,
        eval_dataset=dataset,
    )

    trainer.train()
    torch.save(
        model.soft_tokens.detach().cpu(),
        f"{model_args.model_name_or_path.replace('/', '.')}.soft_prompt.pt")

    trainer.generate_completions()

if __name__ == '__main__':
    # Train the model
    train_prompt_tuning()
