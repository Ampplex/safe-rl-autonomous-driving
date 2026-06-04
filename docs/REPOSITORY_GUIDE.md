# Repository Guide

Suggested GitHub repository name:

```text
safe-rl-autonomous-driving
```

## Layout

```text
configs/          Hyperparameters and safety penalty configuration
environments/     HighwayEnv factory and SafeRewardWrapper
training/         PPO and Safe PPO training scripts
evaluation/       Evaluation, multi-seed, behavior, and statistics scripts
experiments/      Experiment-specific entry points
visualization/    Plot generation
models/           Small trained model checkpoints used for evaluation
results/          Curated CSVs and generated plots
docs/             Reproducibility and result summaries
```

## Files To Keep Out Of Git

The local virtual environment and runtime tracking files should not be committed:

```text
venv/
mlruns/
mlflow.db
logs/
__pycache__/
.DS_Store
SESSION_RESUME.md
```

These are covered by `.gitignore`.

## Files Worth Including

The trained models are small enough to include and make the repo easy to evaluate:

```text
models/*.zip
```

The curated plots and CSVs in `results/` are also worth including because they let reviewers inspect conclusions without rerunning every experiment.

## Recommended GitHub Description

```text
Safe RL for autonomous driving: PPO vs reward-shaped Safe PPO in HighwayEnv with multi-seed validation, statistical tests, and robustness analysis.
```
