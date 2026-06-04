#!/usr/bin/env bash
set -euo pipefail

python evaluation/lambda_multi_seed_eval.py --episodes_per_seed 20 --density 50
python evaluation/statistical_significance.py
python evaluation/lambda_behavior_eval.py
python evaluation/reward_breakdown_eval.py --episodes 50 --density 50
python visualization/generate_plots.py
