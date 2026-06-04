# Reproducibility

Recommended repository name:

```text
safe-rl-autonomous-driving
```

## Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Smoke Test

```bash
python test_env.py
```

## Train Policies

```bash
python training/train_ppo.py
python training/train_safeppo.py --safety_lambda 0.1
python training/train_safeppo.py --safety_lambda 0.5
python training/train_safeppo.py --safety_lambda 1.0
python training/train_safeppo.py --safety_lambda 2.0
```

## Core Evaluations

```bash
python experiments/exp1_comparison.py
python experiments/exp2_generalization.py
python experiments/exp3_ablation.py
python evaluation/lambda_multi_seed_eval.py --episodes_per_seed 20 --density 50
python evaluation/statistical_significance.py
python evaluation/lambda_behavior_eval.py
python evaluation/reward_breakdown_eval.py --episodes 50 --density 50
python experiments/exp5_ood.py
```

## Regenerate Figures

```bash
python visualization/generate_plots.py
```

Primary outputs:

- `results/lambda_multi_seed_summary.csv`
- `results/statistical_significance.csv`
- `results/reward_component_breakdown.csv`
- `results/ood_results.csv`
- `results/plots/`

## Current Caveat

The completed deterministic multi-seed benchmark at density 50 did not find a statistically significant advantage for `lambda=0.1` over baseline PPO. The earlier single-run result remains useful as an exploratory signal, but final claims should use the multi-seed result.
