# Enforce LLMs to follow rules

With recent advancements in Large Language Models (LLMs), LLMs are optimized for general-purpose tasks. However, LLMs are not guaranteed to follow rules of the task. Prompt engineering, which is to curate the prompts for the tasks, is shown to improve rate of rule following only by a small margin. In this report, we demonstrate that our method of curriculum method can effectively increase the rate of rule following across different LLM models. In particular, two-step curriculum learning where each step involves proximal policy optimization and low-rank adaptation is the most effective. For simpler tasks, using one step of proximal policy optimization and low-rank adaptation is sufficient. We also demonstrate that our approach of combining proximal policy optimization and low-rank adaptation requires the right values of hyper-parameters to achieve high rates of rule following.

# Prerequisites

1. Install the necessary Python libraries.

```bash
pip install -r requirements.txt
```

2. All training and testing scripts are run using [SLURM workload manager](https://slurm.schedmd.com/documentation.html). I ran the experiments on SoC Compute Clusters. The cluster must have the following GPUs:
  - `h100-47`: used to run tests of rule-following
  - `h100-96`: used to run fine-tuning experiments (PPO/prompt tuning/curriculum learning)

# Directory structure

Script entry points are in [`scripts/`](#scripts) folder.

## [evaluate](./evaluate/)

This is the experiment on the quality of LLM outputs. The experiment is used to evaluate how far the quality of outputs changes after the LLM is fine-tuned to follow rules. Note that this experiment is only run on the game of ultimate tic-tac-toe.

FYI, the actual source code for the script to evaluate the LLM for this experiment is in [nknguyenhc/ultimate-tictactoe/tree/fyp](https://github.com/nknguyenhc/ultimate-tictactoe/tree/fyp). The compiled JAR file has been committed to this directory.

To run one experiment,

1. Run a test script with [`scripts/test.slurm`](#testslurm) for the game of ultimate tic-tac-toe, before or after fine-tuning. Obtain the output log (as indicated in `--output` SBATCH parameter) and put the log in this directory, renaming it to remove the `result.` prefix and `.txt` suffix, e.g. rename `result.LiquidAI.LFM2-350M.txt` to `LiquidAI.LFM2-350M`.
2. In `evaluate.slurm`, Edit the last argument of the bash command to point to the output log you have just put in the current directory, e.g. `LiquidAI.LFM2-350M`.
3. Optionally, update the SBATCH parameters `--output` and `--error`. This will be the file containing stdout and stderr of this evaluation script.
4. Send the batch script.

```bash
sbatch evaluate.slurm
```

The result of evaluation is then stored in `result.{model name}.txt`, e.g. `result.LiquidAI.LFM2-350M.txt`.

## [fine_tuning](./fine_tuning/)

This folder contains critical components of fine-tuning processes (PPO/prompt tuning/curriculum learning)

- [`c_dataset.py`](./fine_tuning/c_dataset.py): dataset for connect-4 game, for PPO
- [`c_prompt_tuning_dataset.py`](./fine_tuning/c_prompt_tuning_dataset.py): dataset for connect-4 game, for prompt tuning
- [`c_reward.py`](./fine_tuning/c_reward.py): reward model for connect-4, for both PPO and prompt tuning
- [`cc_dataset.py`](./fine_tuning/cc_dataset.py): dataset for xiangqi game, for PPO and final step of curriculum learning
- [`cc_piece_movement_dataset.py`](./fine_tuning/cc_piece_movement_dataset.py): dataset for xiangqi game, for piece movement step curriculum learning
- [`cc_reward.py`](./fine_tuning/cc_reward.py): reward model for xiangqi game, for PPO and final step and piece movement step of curriculum learning
- [`cc_valid_start_dataset.py`](./fine_tuning/cc_valid_start_dataset.py): dataset for xiangqi game, for valid starting position step of curriculum learning
- [`cc_valid_start_reward.py`](./fine_tuning/cc_valid_start_reward.py): reward model for xiangqi game, for valid starting position step of curriculum learning
- [`ult_ttt_dataset.py`](./fine_tuning/ult_ttt_dataset.py): dataset for ultimate tic-tac-toe game, for PPO
- [`ult_ttt_prompt_tuning_dataset.py`](./fine_tuning/ult_ttt_prompt_tuning_dataset.py): dataset for ultimate tic-tac-toe game, for prompt tuning
- [`ult_ttt_reward.py`](./fine_tuning/ult_ttt_reward.py): reward model for ultimate tic-tac-toe game, for both PPO and prompt tuning

## [games](./games/)

This folder contains code for game logic, and the accompanying test scripts (using `unittest` module).

- Connect-4:
  - [`connect_4.py`](./games/connect_4.py): Main logic for the game
  - [`connect_4_config.py`](./games/connect_4_config.py): Configuration for this game (width, height, steal rules, how many pieces in a row to win). Note that the experiments have only been carried out for the current configuration indicated.
  - [`connect_4_test.py`](./games/connect_4_test.py): Test cases for this game
- Tic-tac-toe:
  - [`ttt.py`](./games/ttt.py): Main logic for this game
- Ultimate tic-tac-toe:
  - [`ult_ttt.py`](./games/ult_ttt.py): Main logic for this game
  - [`ult_ttt_test.py`](./games/ult_ttt_test.py): Test cases for this game
- Xiangqi:
  - [`xiangqi.py`](./games/xiangqi.py): Main logic for this game
  - [`xiangqi_test.py`](./games/xiangqi_test.py): Test cases for this game

## [misc](./misc/)

This script contains miscellaneous scripts to analyse training results and plot graphs found in my FYP report.

- [`analysis.py`](./misc/analysis.py): Run analysis and plots graphs from a run log of fine-tuning (e.g. `result.ppo.meta-llama.Llama-3.1-8B-Instruct.out`)
- [`cc_stage_graph.py`](./misc/cc_stage_graph.py): Plot analysis graphs of the 3-stage curricula
- [`overall.py`](./misc/overall.py): Plot performance graphs of different methods on each game

## [scripts](./scripts/)

This folder contains various entry points for fine-tuning and rule following tests.

### [cl.slurm](./scripts/cl.slurm)

This is our main experiment, which is on curriculum learning. The script is used to run one step within a curriculum. Hence to run a full curriculum, you need to run this script multiple times.


1. Update the `output_dir` parameter. This will be the name of the folder containing the model. If the folder is not yet created, the script will create a new folder.
2. Update the `model_name_or_path` parameter to the model that you want to fine-tune on. For the first step, this must be an available model on hugging face, e.g. `LiquidAI/LFM2-350M` points to [https://huggingface.co/LiquidAI/LFM2-350M](https://huggingface.co/LiquidAI/LFM2-350M). For the subsequent step, this must point to a local directory containing the model from the previous training step.
3. Update the `step` parameter to indicate the training step.
  - Use `vls` to train on valid starting positions.
  - Use `pm` to train on piece movement.
  - Use `final` to train on generating full moves.
4. Optionally, update the SBATCH parameters `--output` and `--error`. This will be the file containing stdout and stderr of the training script.
5. Send the batch script.

```bash
sbatch scripts/cl.slurm
```

After each step, there are two tests being run:

1. Testing on valid starting positions and valid moves. The result is stored in `result.valid_start.{output_dir}.txt`.
2. Testing on piece movement. The result is stored in `result.piece_movement.{output_dir}.txt`.

### [ppo.slurm](./scripts/ppo.slurm)

This is the experiment of running PPO with LoRA on ultimate tic-tac-toe, connect-4 or xiangqi.

1. Update the `game` parameter to the game that you want to test on (`ult-ttt`, `connect-4` or `xiangqi`).
2. Update the `model_name_or_path` parameter to the model that you want to fine-tune on. This model must be available on hugging face, e.g. `meta-llama/Llama-3.1-8B-Instruct` points to [https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct). You may use the current value indicated in the script to get started.
3. Update the `trust_remote_code` to indicate the boolean value when loading the LLM. Use `True` for all models, and use `False` for models from `microsoft`, e.g. `microsoft/Phi-3-mini-4k-Instruct` and `microsoft/phi-4`.
4. Optionally, update the `output_dir` parameter. This will be the name of the folder containing the model. If the folder is not yet created, the script will create a new folder.
5. Optionally, update the SBATCH parameters `--output` and `--error`. This will be the file containing stdout and stderr of the training script.
6. Send the batch script.

```bash
sbatch script.slurm
```

After the training script has run, the model will be saved to the folder indicated in `output_dir`. Run the test script with [`test.slurm`](#testslurm) on this fine-tuned model.

### [prompt_tuning.slurm](./scripts/prompt_tuning.slurm)

This is the experiment of running PPO with prefix tuning on ultimate tic-tac-toe and connect-4.

1. Update the `game` parameter to the game that you want to test on (`ult-ttt` or `connect-4`).
2. Update the `model_name_or_path` parameter to the model that you want to fine-tune on. This model must be available on hugging face, e.g. `google/gemma-2-2b-it` points to [https://huggingface.co/google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it). You may use the current value indicated in the script to get started.
3. Update the `trust_remote_code` to indicate the boolean value when loading the LLM. Use `True` for all models, and use `False` for models from `microsoft`, e.g. `microsoft/Phi-3-mini-4k-Instruct` and `microsoft/phi-4`.
4. Optionally, update the `output_dir` parameter. This will be the name of the folder containing the model. If the folder is not yet created, the script will create a new folder.
5. Optionally, update the SBATCH parameters `--output` and `--error`. This will be the file containing stdout and stderr of the training script.
6. Send the batch script.

```bash
sbatch script.slurm
```

After the training script has run, the model will be saved to the folder indicated in `output_dir`. Run the test script with [`test.slurm`](#testslurm) on this fine-tuned model.

### [test.slurm](./scripts/test.slurm)

This is the entry point for the various test scripts. Note that result of the test is stored in `result.{normalized model name}.txt`. Normalized model name is the model name with `/` replaced by `.`, removing the extra `.`'s where necessary, e.g. result of the test on `./google.gemma-2-2b-it` is stored in `result.google.gemma-2-2b-it.txt`.

1. Tic-tac-toe

To run a test script on the game of tic-tac-toe, edit the entry point command to:

```bash
python scripts/test.py \
    --model_name_or_path <model name> \
    --trust_remote_code <True/False> \
    --game ttt
```

Replacing

- `<model name>` with a model available on hugging face
- `<True/False>` with the actual value, use `True` for all models, `False` for models from `microsoft`, e.g. `microsoft/Phi-3-mini-4k-Instruct` and `microsoft/phi-4`.

2. Ultimate tic-tac-toe and connect-4

To run a test script on the game of ultimate tic-tac-toe or connect-4, edit the entry point command to:

```bash
python scripts/test.py \
    --mode <ppo/prompt-tuning> \
    --model_name_or_path <model name> \
    --trust_remote_code <True/False> \
    --game <ult-ttt/connect-4>
```

Replacing

- `<ppo/prompt-tuning>` with the actual mode. Use `ppo` if testing a model from hugging face or a model after PPO. Use `prompt-tuning` if testing a model after prompt tuning. Note that prompt tuning testing requires loading the model in a different way, hence the split in mode.
- `<model name>` with a model available on hugging face, or a fine-tuned model available in local directory, e.g. `./google.gemma-2-2b-it`.
- `<True/False>` with the actual value, use `True` for all models, `False` for models from `microsoft`, e.g. `microsoft/Phi-3-mini-4k-Instruct` and `microsoft/phi-4`.
- `<ult-ttt/connect-4>` with the actual game to test the model on. Use `ult-ttt` for ultimate tic-tac-toe, use `connect-4` for connect-4.

3. Xiangqi

To run a test script on the game of xiangqi, edit the entry point command to:

```bash
python scripts/test.py \
    --step <vls/pm> \
    --model_name_or_path <model name> \
    --trust_remote_code <True/False> \
    --game xiangqi
```

Replacing

- `<vls/pm>` with the metric to test the LLM on. Use `vls` to test for valid starting positions, this can also be used to the model before fine-tuning. Use `pm` to test for piece movement.
- `<model name>` with a model available on hugging face, or a fine-tuned model available in local directory, e.g. `./google.gemma-2-2b-it`.
- `<True/False>` with the actual value, use `True` for all models, `False` for models from `microsoft`, e.g. `microsoft/Phi-3-mini-4k-Instruct` and `microsoft/phi-4`.
