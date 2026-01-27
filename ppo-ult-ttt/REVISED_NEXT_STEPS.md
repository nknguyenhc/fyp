# Revised Next Steps - Valid PPOConfig Parameters Only

## Analysis Summary
- **Entropy Collapse**: 4.1 → 2.4 (model losing exploration)
- **Stagnant Scores**: -2.38 with slope ≈ 0 (no improvement)
- **High KL Divergence**: >2.0 (policy changing too fast vs. reference)
- **Current LR**: 3e-4

## Valid PPOConfig Parameters We Can Tune

Based on the actual PPOConfig API, here are the parameters we CAN adjust:

### Critical Parameters:
1. **`num_ppo_epochs`** (current: 1, default: 4) - How many times to process each batch
2. **`kl_coef`** (default: 0.05) - Penalty for KL divergence from reference policy
3. **`temperature`** (default: 0.7) - Sampling temperature (affects entropy)
4. **`num_mini_batches`** (current: 4, default: 1) - Mini-batch splits
5. **`cliprange`** (default: 0.2) - PPO clipping range
6. **`whiten_rewards`** (default: False) - Normalize rewards

### Other Parameters:
- `vf_coef` (default: 0.1) - Value function loss coefficient
- `gamma` (default: 1.0) - Discount factor
- `lam` (default: 0.95) - GAE lambda

---

## Recommended Changes

### 🔴 Priority 1: Increase PPO Epochs
**Current:** `--num_ppo_epochs 1`  
**Change to:** `--num_ppo_epochs 4`

**Why:** Only 1 epoch means minimal learning from each batch. Default is 4, which is standard for PPO.

---

### 🔴 Priority 2: Increase Temperature (for entropy)
**Current:** Not set (uses default 0.7)  
**Add:** `--temperature 1.0`

**Why:** Higher temperature = more random sampling = higher entropy. This is the key to preventing entropy collapse since there's no `entropy_coef` parameter.

---

### 🟡 Priority 3: Adjust KL Coefficient
**Current:** Not set (uses default 0.05)  
**Add:** `--kl_coef 0.02`

**Why:** Lower KL penalty allows policy to change more. Your current KL is very high (>2.0), suggesting the policy wants to change but is being penalized. Reducing this allows more exploration.

---

### 🟡 Priority 4: Enable Reward Whitening
**Current:** Not set (default False)  
**Add:** `--whiten_rewards True`

**Why:** Normalizes rewards to have mean 0 and std 1, which can stabilize training and improve learning.

---

### 🟢 Priority 5: Increase Clip Range
**Current:** Not set (uses default 0.2)  
**Add:** `--cliprange 0.3`

**Why:** Larger clip range allows bigger policy updates. Current clip fraction is nearly 0, suggesting updates are too conservative.

---

## Recommended Configuration

### Conservative Approach (Try First):
```bash
python ttt_ppo.py \
    --dataset_train_split descriptiveness \
    --response_length 2 \
    --learning_rate 3e-4 \
    --num_ppo_epochs 4 \              # CHANGED: 1 → 4
    --temperature 1.0 \                # NEW: Increase entropy
    --kl_coef 0.02 \                  # NEW: Allow more policy change
    --whiten_rewards True \            # NEW: Stabilize training
    --num_mini_batches 4 \
    --output_dir google.gemma-2-2b-it \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --total_episodes 10000 \
    --trust_remote_code True \
    --model_name_or_path google/gemma-2-2b-it \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target_modules q_proj k_proj v_proj o_proj
```

### Aggressive Approach (If Conservative Fails):
```bash
python ttt_ppo.py \
    --dataset_train_split descriptiveness \
    --response_length 2 \
    --learning_rate 5e-4 \            # INCREASED
    --num_ppo_epochs 6 \              # INCREASED: More training
    --temperature 1.2 \                # INCREASED: More exploration
    --kl_coef 0.01 \                  # DECREASED: Less penalty
    --whiten_rewards True \
    --cliprange 0.3 \                 # INCREASED: Bigger updates
    --num_mini_batches 8 \            # INCREASED: More gradient steps
    --output_dir google.gemma-2-2b-it \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --total_episodes 10000 \
    --trust_remote_code True \
    --model_name_or_path google/gemma-2-2b-it \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target_modules q_proj k_proj v_proj o_proj
```

---

## How These Changes Address the Issues

| Issue | Solution | Parameter |
|-------|----------|-----------|
| Entropy collapse | Higher sampling temperature | `temperature 1.0` → `1.2` |
| Stagnant learning | More training per batch | `num_ppo_epochs 1` → `4+` |
| High KL divergence | Lower KL penalty | `kl_coef 0.05` → `0.02` |
| Conservative updates | Larger clip range | `cliprange 0.2` → `0.3` |
| Unstable rewards | Normalize rewards | `whiten_rewards True` |

---

## Success Metrics

Monitor these after the next run:

| Metric | Current | Target |
|--------|---------|--------|
| Entropy (end) | 2.4 | >3.5 |
| Score Mean | -2.38 | >-2.0 |
| Score Slope | ~0 | >0.0001 |
| KL Mean | 1.0 | 0.2-0.5 |
| Clip Frac | ~0 | 0.05-0.20 |

---

## Implementation Order

1. **First Run - Conservative:**
   - `num_ppo_epochs 4`
   - `temperature 1.0`
   - `kl_coef 0.02`
   - `whiten_rewards True`

2. **If Still Failing - Aggressive:**
   - `num_ppo_epochs 6`
   - `temperature 1.2`
   - `kl_coef 0.01`
   - `cliprange 0.3`
   - `num_mini_batches 8`
   - `learning_rate 5e-4`

---

## Additional Notes

### Why No Entropy Coefficient?
TRL's PPOConfig doesn't have an explicit `entropy_coef` parameter. Entropy is controlled through:
1. **Temperature** - Primary control for sampling entropy
2. **Model's inherent loss function** - May have built-in entropy regularization

### Temperature vs Entropy
- `temperature 0.7` (default) = Less random, lower entropy
- `temperature 1.0` = Standard sampling, balanced entropy
- `temperature 1.2+` = More random, higher entropy

### KL Coefficient Tradeoff
- Higher `kl_coef` = Policy stays closer to reference (conservative)
- Lower `kl_coef` = Policy can deviate more (exploratory)
- Your high KL (>2.0) suggests lowering this will help

# Quick Reference: Valid Parameter Changes

## Current Configuration (Failing)
```bash
--learning_rate 3e-4
--num_ppo_epochs 1              # Too low
--num_mini_batches 4
# temperature: defaults to 0.7
# kl_coef: defaults to 0.05
# whiten_rewards: defaults to False
```

**Results:** Entropy collapsed, scores stagnant at -2.38

---

## Conservative Fix (Try First)
```bash
--learning_rate 3e-4            # Keep same
--num_ppo_epochs 4              # 1 → 4 (4x more training)
--temperature 1.0               # 0.7 → 1.0 (more entropy)
--kl_coef 0.02                  # 0.05 → 0.02 (allow more change)
--whiten_rewards True           # Normalize rewards
--num_mini_batches 4            # Keep same
```

**File:** `script_conservative_v2.slurm`

---

## Aggressive Fix (Backup)
```bash
--learning_rate 5e-4            # 3e-4 → 5e-4 (faster learning)
--num_ppo_epochs 6              # 1 → 6 (6x more training)
--temperature 1.2               # 0.7 → 1.2 (strong exploration)
--kl_coef 0.01                  # 0.05 → 0.01 (minimal penalty)
--whiten_rewards True           # Normalize rewards
--cliprange 0.3                 # 0.2 → 0.3 (bigger updates)
--num_mini_batches 8            # 4 → 8 (more gradient steps)
```

**File:** `script_aggressive_v2.slurm`

---

## What Each Parameter Does

| Parameter | Effect on Training | Why Change It |
|-----------|-------------------|---------------|
| `num_ppo_epochs` | More epochs = more learning per batch | Currently only 1 (too low) |
| `temperature` | Higher = more random = higher entropy | Fix entropy collapse |
| `kl_coef` | Lower = allow more policy deviation | High KL suggests too restrictive |
| `whiten_rewards` | Normalizes rewards | Stabilizes learning |
| `cliprange` | Higher = allows bigger updates | Current updates too small |
| `num_mini_batches` | More batches = more gradient steps | Improves stability |

---

## Submit Command

```bash
# Try conservative first
sbatch script_conservative_v2.slurm

# If that doesn't work, try aggressive
sbatch script_aggressive_v2.slurm
```

---

## Monitor Progress

```bash
# Check latest entropy values
tail -n 100 result.ppo.conservative.gemma-2-2b-it.out | grep "objective/entropy"

# Should stay above 3.5 instead of dropping to 2.4
```
