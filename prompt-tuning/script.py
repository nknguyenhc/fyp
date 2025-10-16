import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import sys

from ttt_dataset import get_dataset

# 1. DATASET STRUCTURE
# Each example should have:
# - 'query': the text prompt/question
# - 'output': the expected completion/answer

dataset = [
    {"query": "Translate to French: Hello", "output": "Bonjour"},
    {"query": "Translate to French: Goodbye", "output": "Au revoir"},
    {"query": "Translate to French: Thank you", "output": "Merci"},
    {"query": "Translate to French: Good morning", "output": "Bonjour"},
    {"query": "Translate to French: How are you?", "output": "Comment allez-vous?"},
    {"query": "Sentiment: I love this movie!", "output": "Positive"},
    {"query": "Sentiment: This is terrible.", "output": "Negative"},
    {"query": "Sentiment: It was okay.", "output": "Neutral"},
]

class PromptTuningDataset(Dataset):
    """Dataset class for prompt tuning"""
    def __init__(self, data, tokenizer):
        texts = [f"{item['query']} {item['output']}" for item in data]
        encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        self.data = [{
            'input': item['query'],
            'output': item['output'],
            'input_ids': encodings['input_ids'][i],
            'attention_mask': encodings['attention_mask'][i],
            'labels': encodings['input_ids'][i].clone(),
        } for i, item in enumerate(data)]  # Store original input/output for reference
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

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
    
    def forward(self, input_ids, attention_mask=None, labels=None):
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
        
        # Adjust labels to account for soft prompt
        if labels is not None:
            # Pad labels with -100 (ignore index) for soft prompt positions
            prompt_labels = torch.full(
                (batch_size, self.n_prompt_tokens),
                -100,
                dtype=labels.dtype,
                device=labels.device
            )
            labels = torch.cat([prompt_labels, labels], dim=1)
        
        # Forward pass through model
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs
    
    def generate(self, input_ids, attention_mask, max_new_tokens=2):
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
        
        # Generate
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens
        )
        return outputs

def train_prompt_tuning(
    model_name: str,
    n_prompt_tokens=20,
    epochs=1,
    batch_size=2,
    lr=3e-6,
):
    """Train prompt tuning"""
    
    # Load tokenizer and base model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Create prompt tuning model
    model = PromptTuning(base_model, n_prompt_tokens)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Prepare dataset
    dataset = get_dataset()
    train_dataset = PromptTuningDataset(dataset, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer only for soft prompt parameters
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr
    )
    
    # Training loop
    print("\nStarting training...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')
    
    print("\nTraining completed!")
    return model, tokenizer

def generate_with_prompt_tuning(model, tokenizer, prompt, max_length=50):
    """Generate text using the trained soft prompt"""
    model.eval()
    device = next(model.parameters()).device
    
    # Tokenize input
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    
    with torch.no_grad():
        # Get embeddings with soft prompt
        inputs_embeds = model.model.get_input_embeddings()(input_ids)
        soft_prompt = model.soft_prompt.unsqueeze(0)
        inputs_embeds = torch.cat([soft_prompt, inputs_embeds], dim=1)
        
        # Generate
        outputs = model.model.generate(
            inputs_embeds=inputs_embeds,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == '__main__':
    # Train the model
    model_name = sys.argv[1]
    model, tokenizer = train_prompt_tuning(
        model_name=model_name,
    )

    # Save model
    model_save_path = f"{model_name.replace('/', '_')}_prompt_tuned.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
