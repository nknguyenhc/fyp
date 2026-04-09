# Enforce LLMs to follow rules

With recent advancements in Large Language Models (LLMs), LLMs are optimized for general-purpose tasks. However, LLMs are not guaranteed to follow rules of the task. Prompt engineering, which is to curate the prompts for the tasks, is shown to improve rate of rule following only by a small margin. In this report, we demonstrate that our method of curriculum method can effectively increase the rate of rule following across different LLM models. In particular, two-step curriculum learning where each step involves proximal policy optimization and low-rank adaptation is the most effective. For simpler tasks, using one step of proximal policy optimization and low-rank adaptation is sufficient. We also demonstrate that our approach of combining proximal policy optimization and low-rank adaptation requires the right values of hyper-parameters to achieve high rates of rule following.

# Prerequisites

1. Install the necessary Python libraries.

```bash
pip install -r requirements.txt
```

2. All training and testing scripts are run using [SLURM workload manager](https://slurm.schedmd.com/documentation.html). I ran the experiments on SoC Compute Clusters. The cluster must have the following GPUs:
  - `h100-47`
  - `h100-96`

# Directory structure

This repository contains all of my experiments, each in a separate folder. Due to time constraint, some codes, such as game rules and prompts, are copy-pasted across a few folders. Below are the folder names and the description of the corresponding experiments.

Note that there are side experiments that are not included in our final report, and will not be described here.

## cl-xiangqi

This is our main experiment, which is on curriculum learning. There are two curriculum-learning experiments being explored here:

- The first curriculum learning approach is to train on valid starting position, then train on generating a full move. The entry point of this curriculum is `valid_start.slurm`.
- The second curriculum learning approach is to train on piece movement, then train on generating a full move. The entry point of this curriculum is `piece_movement.slurm`.

To start a curriculum learning, open the entry point script (`valid_start.slurm` or `piece_movement.slurm`):

1. Update the flags within the corresponding entry point Python script (`valid_start.py` or `piece_movement.py`), `FIRST_SPEP` and `SECOND_STEP` flags. These flags indicate whether to perform first step or second step of the curriculum respectively.
2. Update the `model_name_or_path` parameter to the model that you want to fine-tune on. This model must be available on hugging face, e.g. `LiquidAI/LFM2-350M` points to [https://huggingface.co/LiquidAI/LFM2-350M](https://huggingface.co/LiquidAI/LFM2-350M), or it must point to a local directory containing the model. You may use the current value indicated in the script to get started.
3. Optionally, update the `output_dir` parameter. This will be the name of the folder containing the model. If the folder is not yet created, the script will create a new folder.
4. Optionally, update the SBATCH parameters `--output` and `--error`. This will be the file containing stdout and stderr of the training script.
5. Send the batch script.

```bash
sbatch valid_start.slurm
```

OR

```bash
sbatch piece_movement.slurm
```

Normally, I would send 2 jobs for each curriculum learning.

1. In the first step, set the following flags in the Python script:

```py
FIRST_STEP = True
SECOND_STEP = False
```

And set `--output_dir vls.LiquidAI.LFM2-350M` and `--model_name_or_path LiquidAI/LFM2-350M`. This runs the first step in the curriculum from a model available on hugging face.

2. In the second step, set the following flags in the Python script:

```py
FIRST_STEP = False
SECOND_STEP = True
```

And set `--model_name_or_path vls.LiquidAI.LFM2-350M`, which points to the fine-tuned model from step 1, and `--output_dir vls2.LiquidAI.LFM2-350M`. This runs the second step in the curriculum from the model fine-tuned in step 1.

The evaluation result of the fine-tuned model is then stored in `result.{output_dir}.txt`, e.g. `result.vls2.LiquidAI.LFM2-350M`.

## cl2-xiangqi

This is our experiment on the third and fourth curriculum learning approaches, which have shown inferior performance over our main curriculum learning approaches in [`cl-xiangqi`](#cl-xiangqi).

- The third curriculum learning approach is to train on valid starting position, then train on piece movement, then train on generating a full move.
- The fourth curriculum learning approach is to train on piece movement, then train on valid starting position, then train on generating a full move.

The entry point script `script.slurm` runs one step of the curriculum learning. Hence, to run each curriculum, you need to run the script 3 times sequentially. To run each step, do the following:

1. Update the `output_dir` parameter. This will be the name of the folder containing the model. If the folder is not yet created, the script will create a new folder.
2. Update the `model_name_or_path` parameter to the model that you want to fine-tune on. For the first step, this must be an available model on hugging face, e.g. `LiquidAI/LFM2-350M` points to [https://huggingface.co/LiquidAI/LFM2-350M](https://huggingface.co/LiquidAI/LFM2-350M). For the subsequent step, this must point to a local directory containing the model from the previous training step.
3. Update the `step` parameter to indicate the training step.
  - Use `vls` to train on valid starting positions.
  - Use `pm` to train on piece movement.
  - Use `final` to train on generating full moves.
4. Optionally, update the SBATCH parameters `--output` and `--error`. This will be the file containing stdout and stderr of the training script.
5. Send the batch script.

```bash
sbatch script.slurm
```

After each step, there are two tests being run:

1. Testing on valid starting positions and valid moves. The result is stored in `result.valid_start.{output_dir}.txt`.
2. Testing on piece movement. The result is stored in `result.piece_movement.{output_dir}.txt`.

## connect-4

This is the experiment on rule following rates of LLMs for the game of connect-4. To start a test, in `script.slurm`,

1. Update the third argument within the last line to point to a model to test. This must point to an available model on hugging face, e.g. `google/gemma-2-2b-it` points to [https://huggingface.co/google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it).
2. Update the fourth argument to indicate the boolean value for `trust_remote_code` when loading the LLM. Use `true` for all models, and use `false` for models from `microsoft`, e.g. `microsoft/Phi-3-mini-4k-Instruct` and `microsoft/phi-4`.
3. Optionally, update the SBATCH parameters `--output` and `--error`. This will be the file containing stdout and stderr of the training script.
4. Send the batch script.

```bash
sbatch script.slurm
```

The result of the test is then stored in `result.{normalized model name}.txt`. Normalized model name is the model name with `/` replaced by `.`, e.g. result of the test on `google/gemma-2-2b-it` is stored in `result.google.gemma-2-2b-it.txt`.

## evaluate

This is the experiment on the quality of LLM outputs. The experiment is used to evaluate how far the quality of outputs changes after the LLM is fine-tuned to follow rules. Note that this experiment is only run on the game of ultimate tic-tac-toe.

FYI, the actual source code for the script to evaluate the LLM for this experiment is in [nknguyenhc/ultimate-tictactoe/tree/fyp](https://github.com/nknguyenhc/ultimate-tictactoe/tree/fyp). The compiled JAR file has been committed to this directory.

To run one experiment,

1. Run a test script in [`ppo-ult-ttt`](#ppo-ult-ttt). Obtain the output log (as indicated in `--output` SBATCH parameter) and put the log in this directory, renaming it to remove the `result.` prefix and `.txt` suffix, e.g. rename `result.LiquidAI.LFM2-350M.txt` to `LiquidAI.LFM2-350M`.
2. In `evaluate.slurm`, Edit the last argument of the bash command to point to the output log you have just put in the current directory, e.g. `LiquidAI.LFM2-350M`.
3. Optionally, update the SBATCH parameters `--output` and `--error`. This will be the file containing stdout and stderr of this evaluation script.
4. Send the batch script.

```bash
sbatch evaluate.slurm
```

The result of evaluation is then stored in `result.{model name}.txt`, e.g. `result.LiquidAI.LFM2-350M.txt`.

## ppo-connect-4

This is the experiment of running PPO with LoRA on the game of connect-4. Do the following steps in `script.slurm`.

1. Update the `model_name_or_path` parameter to the model that you want to fine-tune on. This model must be available on hugging face, e.g. `LiquidAI/LFM2-350M` points to [https://huggingface.co/LiquidAI/LFM2-350M](https://huggingface.co/LiquidAI/LFM2-350M). You may use the current value indicated in the script to get started.
2. Update the `trust_remote_code` to indicate the boolean value when loading the LLM. Use `True` for all models, and use `False` for models from `microsoft`, e.g. `microsoft/Phi-3-mini-4k-Instruct` and `microsoft/phi-4`.
3. Optionally, update the `output_dir` parameter. This will be the name of the folder containing the model. If the folder is not yet created, the script will create a new folder.
4. Optionally, update the SBATCH parameters `--output` and `--error`. This will be the file containing stdout and stderr of the training script.
5. Send the batch script.

```bash
sbatch script.slurm
```

After the training script has run, the model will be saved to the folder indicated in `output_dir`. Run the test script on this fine-tuned model.

1. Update the third argument within the last line to point to the output directory from the training script to test, e.g. `./google.gemma-2-2b-it`.
2. Update the fourth argument to indicate the boolean value for `trust_remote_code` when loading the LLM. Use `true` for all models, and use `false` for models from `microsoft`, e.g. `./microsoft.Phi-3-mini-4k-Instruct` and `./microsoft.phi-4`.
3. Optionally, update the SBATCH parameters `--output` and `--error`. This will be the file containing stdout and stderr of the training script.
4. Send the batch script.

```bash
sbatch test.slurm
```

The result of the test on the fine-tuned model is then stored in `result.{normalized model name}.txt`. Normalized model name is the model name with `/` replaced by `.`, removing the extra `.`'s where necessary, e.g. result of the test on `./google.gemma-2-2b-it` is stored in `result.google.gemma-2-2b-it.txt`.

## ppo-ult-ttt

This is the experiment of running PPO with LoRA on the game of ultimate tic-tac-toe. Do the following steps in `script.slurm`.

1. Update the `model_name_or_path` parameter to the model that you want to fine-tune on. This model must be available on hugging face, e.g. `meta-llama/Llama-3.1-8B-Instruct` points to [https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct). You may use the current value indicated in the script to get started.
2. Update the `trust_remote_code` to indicate the boolean value when loading the LLM. Use `True` for all models, and use `False` for models from `microsoft`, e.g. `microsoft/Phi-3-mini-4k-Instruct` and `microsoft/phi-4`.
3. Optionally, update the `output_dir` parameter. This will be the name of the folder containing the model. If the folder is not yet created, the script will create a new folder.
4. Optionally, update the SBATCH parameters `--output` and `--error`. This will be the file containing stdout and stderr of the training script.
5. Send the batch script.

```bash
sbatch script.slurm
```

After the training script has run, the model will be saved to the folder indicated in `output_dir`. Run the test script on this fine-tuned model.

1. Update the third argument within the last line to point to the output directory from the training script to test, e.g. `./google.gemma-2-2b-it`.
2. Update the fourth argument to indicate the boolean value for `trust_remote_code` when loading the LLM. Use `true` for all models, and use `false` for models from `microsoft`, e.g. `./microsoft.Phi-3-mini-4k-Instruct` and `./microsoft.phi-4`.
3. Optionally, update the SBATCH parameters `--output` and `--error`. This will be the file containing stdout and stderr of the training script.
4. Send the batch script.

```bash
sbatch test.slurm
```

The result of the test on the fine-tuned model is then stored in `result.{normalized model name}.txt`. Normalized model name is the model name with `/` replaced by `.`, removing the extra `.`'s where necessary, e.g. result of the test on `./google.gemma-2-2b-it` is stored in `result.google.gemma-2-2b-it.txt`.

## ppo-xiangqi

This is the experiment of running PPO with LoRA on the game of xiangqi. Do the following steps in `script.slurm`.

1. Update the `model_name_or_path` parameter to the model that you want to fine-tune on. This model must be available on hugging face, e.g. `LiquidAI/LFM2-350M` points to [https://huggingface.co/LiquidAI/LFM2-350M](https://huggingface.co/LiquidAI/LFM2-350M). You may use the current value indicated in the script to get started.
2. Update the `trust_remote_code` to indicate the boolean value when loading the LLM. Use `True` for all models, and use `False` for models from `microsoft`, e.g. `microsoft/Phi-3-mini-4k-Instruct` and `microsoft/phi-4`.
3. Update the `response_length` to fit the model.
  * Use `3` for `LiquidAI/LFM2-350M`, `meta-llama/Llama-3.1-8B-Instruct`, `openai/gpt-oss-20b`.
  * Use `5` for `google/gemma-2-2b-it`, `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen3-8B`
4. Optionally, update the `output_dir` parameter. This will be the name of the folder containing the model. If the folder is not yet created, the script will create a new folder.
5. Optionally, update the SBATCH parameters `--output` and `--error`. This will be the file containing stdout and stderr of the training script.
6. Send the batch script.

```bash
sbatch script.slurm
```

After the training script has run, the model will be saved to the folder indicated in `output_dir`. Run the test script on this fine-tuned model.

1. Update the third argument within the last line to point to the output directory from the training script to test, e.g. `./google.gemma-2-2b-it`.
2. Update the fourth argument to indicate the boolean value for `trust_remote_code` when loading the LLM. Use `true` for all models, and use `false` for models from `microsoft`, e.g. `./microsoft.Phi-3-mini-4k-Instruct` and `./microsoft.phi-4`.
3. Optionally, update the SBATCH parameters `--output` and `--error`. This will be the file containing stdout and stderr of the training script.
4. Send the batch script.

```bash
sbatch test.slurm
```

The result of the test on the fine-tuned model is then stored in `result.{normalized model name}.txt`. Normalized model name is the model name with `/` replaced by `.`, removing the extra `.`'s where necessary, e.g. result of the test on `./google.gemma-2-2b-it` is stored in `result.google.gemma-2-2b-it.txt`.

## prompt-tuning

This is the experiment of running PPO with prefix tuning on the game of ultimate tic-tac-toe. Do the following steps in `script.slurm`.

1. Update the `model_name_or_path` parameter to the model that you want to fine-tune on. This model must be available on hugging face, e.g. `google/gemma-2-2b-it` points to [https://huggingface.co/google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it). You may use the current value indicated in the script to get started.
2. Update the `trust_remote_code` to indicate the boolean value when loading the LLM. Use `True` for all models, and use `False` for models from `microsoft`, e.g. `microsoft/Phi-3-mini-4k-Instruct` and `microsoft/phi-4`.
3. Optionally, update the `output_dir` parameter. This will be the name of the folder containing the model. If the folder is not yet created, the script will create a new folder.
4. Optionally, update the SBATCH parameters `--output` and `--error`. This will be the file containing stdout and stderr of the training script.
5. Send the batch script.

```bash
sbatch script.slurm
```

After the training script has run, the model will be saved to the folder indicated in `output_dir`. Run the test script on this fine-tuned model.

1. Update the third argument within the last line to point to the output directory from the training script to test, e.g. `./google.gemma-2-2b-it`.
2. Update the fourth argument to indicate the boolean value for `trust_remote_code` when loading the LLM. Use `true` for all models, and use `false` for models from `microsoft`, e.g. `./microsoft.Phi-3-mini-4k-Instruct` and `./microsoft.phi-4`.
3. Optionally, update the SBATCH parameters `--output` and `--error`. This will be the file containing stdout and stderr of the training script.
4. Send the batch script.

```bash
sbatch test.slurm
```

The result of the test on the fine-tuned model is then stored in `result.{normalized model name}.txt`. Normalized model name is the model name with `/` replaced by `.`, removing the extra `.`'s where necessary, e.g. result of the test on `./google.gemma-2-2b-it` is stored in `result.google.gemma-2-2b-it.txt`.

## prompt-tuning-connect-4

This is the experiment of running PPO with prefix tuning on the game of ultimate tic-tac-toe. Do the following steps in `script.slurm`.

1. Update the `model_name_or_path` parameter to the model that you want to fine-tune on. This model must be available on hugging face, e.g. `google/gemma-2-2b-it` points to [https://huggingface.co/google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it). You may use the current value indicated in the script to get started.
2. Update the `trust_remote_code` to indicate the boolean value when loading the LLM. Use `True` for all models, and use `False` for models from `microsoft`, e.g. `microsoft/Phi-3-mini-4k-Instruct` and `microsoft/phi-4`.
3. Optionally, update the `output_dir` parameter. This will be the name of the folder containing the model. If the folder is not yet created, the script will create a new folder.
4. Optionally, update the SBATCH parameters `--output` and `--error`. This will be the file containing stdout and stderr of the training script.
5. Send the batch script.

```bash
sbatch script.slurm
```

After the training script has run, the model will be saved to the folder indicated in `output_dir`. Run the test script on this fine-tuned model.

1. Update the third argument within the last line to point to the output directory from the training script to test, e.g. `./google.gemma-2-2b-it`.
2. Update the fourth argument to indicate the boolean value for `trust_remote_code` when loading the LLM. Use `true` for all models, and use `false` for models from `microsoft`, e.g. `./microsoft.Phi-3-mini-4k-Instruct` and `./microsoft.phi-4`.
3. Optionally, update the SBATCH parameters `--output` and `--error`. This will be the file containing stdout and stderr of the training script.
4. Send the batch script.

```bash
sbatch test.slurm
```

The result of the test on the fine-tuned model is then stored in `result.{normalized model name}.txt`. Normalized model name is the model name with `/` replaced by `.`, removing the extra `.`'s where necessary, e.g. result of the test on `./google.gemma-2-2b-it` is stored in `result.google.gemma-2-2b-it.txt`.

## ult-ttt

This is the experiment on rule following rates of LLMs for the game of ultimate tic-tac-toe. To start a test, in `script.slurm`,

1. Update the third argument within the last line to point to a model to test. This must point to an available model on hugging face, e.g. `google/gemma-2-2b-it` points to [https://huggingface.co/google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it).
2. Update the fourth argument to indicate the boolean value for `trust_remote_code` when loading the LLM. Use `true` for all models, and use `false` for models from `microsoft`, e.g. `microsoft/Phi-3-mini-4k-Instruct` and `microsoft/phi-4`.
3. Optionally, update the SBATCH parameters `--output` and `--error`. This will be the file containing stdout and stderr of the training script.
4. Send the batch script.

```bash
sbatch script.slurm
```

The result of the test is then stored in `result.{normalized model name}.txt`. Normalized model name is the model name with `/` replaced by `.`, e.g. result of the test on `google/gemma-2-2b-it` is stored in `result.google.gemma-2-2b-it.txt`.

## xiangqi

This is the experiment on rule following rates of LLMs for the game of xiangqi. To start a test, in `script.slurm`,

1. Update the third argument within the last line to point to a model to test. This must point to an available model on hugging face, e.g. `google/gemma-2-2b-it` points to [https://huggingface.co/google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it).
2. Update the fourth argument to indicate the boolean value for `trust_remote_code` when loading the LLM. Use `true` for all models, and use `false` for models from `microsoft`, e.g. `microsoft/Phi-3-mini-4k-Instruct` and `microsoft/phi-4`.
3. Optionally, update the SBATCH parameters `--output` and `--error`. This will be the file containing stdout and stderr of the training script.
4. Send the batch script.

```bash
sbatch script.slurm
```

The result of the test is then stored in `result.{normalized model name}.txt`. Normalized model name is the model name with `/` replaced by `.`, e.g. result of the test on `google/gemma-2-2b-it` is stored in `result.google.gemma-2-2b-it.txt`.
