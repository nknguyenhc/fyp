# PPO Training Analysis & Fixes

## 📊 Analysis Summary

**Training Run:** `result.ppo.google.gemma-2-2b-it.out`  
**Episodes:** 20,096 episodes (~20 epochs)  
**Status:** ❌ **No learning progress detected**

### Key Metrics

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| **Objective Scores** | -2.390 | 0.177 | -2.766 | -1.734 |
| **RLHF Reward** | -2.393 | 0.177 | -2.770 | -1.737 |
| **KL Divergence** | 0.044 | 0.037 | -0.019 | 0.165 |
| **Entropy** | 4.109 | 0.112 | 3.836 | 4.454 |
| **Policy Loss** | -0.00009 | 0.0012 | - | - |
| **Value Loss** | 0.590 | 0.034 | - | - |
| **Policy Clipfrac** | 0.00068 | - | 0 | 0.0029 |
| **ApproxKL** | 0.00062 | - | 0 | 0.0020 |

### Learning Progress
- **Score Improvement:** -0.094 (worse!)
- **Start Score:** -2.297 (episode 384)
- **End Score:** -2.391 (episode 20,096)
- **Non-zero clipfrac:** 71/146 batches (48%)

## 🚨 Critical Issues Identified

### 1. **Fatal Reward Function Flaw** ⚠️ CRITICAL

**Problem:** The reward function only checked move validity, not game outcomes!

```python
# BEFORE (WRONG):
if is_valid_action(state, action):
    return 3      # All valid moves got same reward
else:
    return -3
```

**Impact:**
- Model learned to make valid moves but never learned to WIN
- No incentive to pursue winning strategies
- Explains flat learning curve

**Fix Applied:** ✅
```python
# AFTER (CORRECT):
if not is_valid_action(state, action):
    return -3  # Invalid move penalty

new_state = change_state(state, action, check_valid_action=False)

if is_terminal(new_state):
    winner = board_status(new_state.local_board_status)
    if winner == state.fill_num:
        return 10  # Win reward!
    elif winner == 3:
        return 0  # Draw
    else:
        return -10  # Loss
else:
    return 1  # Valid move
```

### 2. **Learning Rate Too Small** ⚠️ HIGH PRIORITY

**Problem:**
- Used: `3e-6` (0.000003)
- Standard LoRA: `3e-4` (0.0003)
- **100x too small!**

**Evidence:**
- Extremely low clipfrac (0.00068 vs target 0.05-0.20)
- Minimal policy updates
- LR decayed to `1.91e-08` by end

**Fix Applied:** ✅ Changed to `3e-4`

### 3. **Insufficient PPO Epochs** ⚠️ MEDIUM PRIORITY

**Problem:**
- Used: `num_ppo_epochs=1`
- Standard: 3-5 epochs
- Single pass insufficient for policy convergence

**Fix Applied:** ✅ Changed to `4`

## ✅ Changes Made

### File: `ttt_reward.py`
- ✅ Added game outcome detection
- ✅ Reward winning moves (+10)
- ✅ Penalize losing moves (-10)
- ✅ Neutral reward for draws (0)
- ✅ Small positive for valid continuing moves (+1)
- ✅ Keep penalties for invalid moves (-3) and parse errors (-9)

### File: `script.slurm`
- ✅ Increased learning rate: `3e-6` → `3e-4` (100x)
- ✅ Increased PPO epochs: `1` → `4` (4x)

## 📋 Next Steps

### Immediate Actions:
1. **Re-run training** with fixed reward function and hyperparameters
2. **Monitor early episodes** (first 500) for:
   - Scores should start improving
   - Clipfrac should increase to 0.05-0.15
   - Positive policy updates

### Expected Improvements:
- **Scores:** Should trend upward toward +10 (wins)
- **Clipfrac:** Should increase to 0.05-0.20 range
- **KL Divergence:** May increase slightly (acceptable)
- **Learning:** Should see meaningful progress within 1000 episodes

### Monitoring Checklist:
- [ ] Scores trending upward
- [ ] Win rate increasing
- [ ] Clipfrac in healthy range (0.05-0.20)
- [ ] Policy loss stable
- [ ] No gradient explosions

### If Still Not Learning:

1. **Add reward shaping:**
   ```python
   # Reward board control/strategic positions
   # Penalize moves that allow opponent wins
   ```

2. **Increase batch size:**
   - Try `per_device_train_batch_size=8`

3. **Adjust PPO clip range:**
   - Add `--clip_range 0.2` explicitly

4. **Verify dataset quality:**
   - Check if training examples are diverse
   - Ensure both win/loss scenarios

## 🔬 Additional Observations

### What Was Working:
- ✅ Value loss stable (~0.59)
- ✅ Entropy maintained (~4.1) - exploration active
- ✅ No training crashes
- ✅ Valid move generation mostly working

### Root Cause:
**The model was optimizing the wrong objective!** It learned to play legally but not strategically, because the reward signal didn't differentiate between winning and losing moves.

## 📚 References

- Standard PPO LR for LoRA: 1e-4 to 5e-4
- Target KL Divergence: 0.01-0.05
- Target Clipfrac: 0.05-0.20
- PPO Epochs: 3-5 for most tasks

---
**Date:** 2026-01-24  
**Status:** Fixed - Ready for re-training
