import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from trl import PPOTrainer, ScriptArguments, PPOConfig, ModelConfig, get_quantization_config, get_kbit_device_map, get_peft_config
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from accelerate import PartialState
import shutil
import gc

from ttt_dataset import get_dataset
from ttt_reward import TTTReward

def prepare_dataset(dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples["query"], padding=True, truncation=True)

    return dataset.map(tokenize_function, batched=True, remove_columns=['query'])

class PromptTuning(nn.Module):
    """
    Prompt Tuning: Learn continuous prompt embeddings while keeping
    the base model frozen.
    """
    def __init__(self, model, n_prompt_tokens):
        super().__init__()
        self.model = model
        self.n_prompt_tokens = n_prompt_tokens
        
        # Freeze all model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Get embedding dimension from the model
        embed_dim = self.model.config.hidden_size
        
        # Initialize learnable soft prompts
        # These are continuous embeddings (not discrete tokens)
        self.soft_prompt = nn.Parameter(
            torch.randn(n_prompt_tokens, embed_dim) * 0.01
        )
        
        print(f"Initialized soft prompt: {self.soft_prompt.shape}")
        print(f"Total trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def forward(self, **kwargs):
        with torch.no_grad():
            return self.model(**kwargs)

    def generate(self, input_ids, attention_mask, **kwargs):
        # Get input embeddings
        batch_size = input_ids.shape[0]
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        
        # Expand soft prompt for batch
        soft_prompt_batch = self.soft_prompt.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        
        # Concatenate soft prompt with input embeddings
        inputs_embeds = torch.cat([soft_prompt_batch, inputs_embeds], dim=1)
        
        # Extend attention mask for soft prompt
        if attention_mask is not None:
            prompt_mask = torch.ones(
                batch_size, self.n_prompt_tokens,
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        
        # Forward pass through model
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs
        )
        outputs.sequences = torch.cat([input_ids, outputs.sequences], dim=1)
        
        return outputs
    
    def __getattr__(self, name):
        # Delegate attribute access to the underlying model
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

def train_prompt_tuning(
    n_prompt_tokens=20,
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
    model_name = model_args.model_name_or_path
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Create prompt tuning model
    model = PromptTuning(base_model, n_prompt_tokens)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Prepare dataset
    dataset = get_dataset()
    with PartialState().local_main_process_first():
        dataset = prepare_dataset(dataset, tokenizer)
    
    peft_config = get_peft_config(model_args)
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
    torch.save(trainer.policy_model.soft_prompt.state_dict(), "soft_prompt.pth")

    trainer.generate_completions()

if __name__ == '__main__':
    # Train the model
    train_prompt_tuning()
