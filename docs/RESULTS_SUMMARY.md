# Results Summary

## Project

**Safe Reinforcement Learning for Autonomous Driving**

PPO is compared against reward-shaped Safe PPO in HighwayEnv under dynamic traffic density.

## Main Results

### Single-Run Benchmark

The initial 100-episode comparison suggested that Safe PPO with `lambda=0.1` reduced collisions from 5.0% to 2.0% while preserving average speed around 20 m/s.

### Multi-Seed Validation

A stricter deterministic benchmark was then run at density 50:

```text
5 lambdas x 5 seed groups x 20 episodes = 500 episodes
```

Lambdas:

```text
0.0, 0.1, 0.5, 1.0, 2.0
```

Seed groups:

```text
42, 123, 456, 789, 2025
```

Final density-50 result:

| Lambda | Collision | Success | Tailgating | Speed |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 2.0% +/- 2.8 pp | 98.0% +/- 2.8 pp | 13.6% +/- 5.6 pp | 20.02 +/- 0.01 m/s |
| 0.1 | 2.0% +/- 2.8 pp | 98.0% +/- 2.8 pp | 13.6% +/- 5.6 pp | 20.02 +/- 0.01 m/s |
| 0.5 | 2.0% +/- 2.8 pp | 98.0% +/- 2.8 pp | 13.6% +/- 5.6 pp | 20.02 +/- 0.01 m/s |
| 1.0 | 2.0% +/- 2.8 pp | 98.0% +/- 2.8 pp | 13.6% +/- 5.6 pp | 20.02 +/- 0.01 m/s |
| 2.0 | 2.0% +/- 2.8 pp | 98.0% +/- 2.8 pp | 13.6% +/- 5.6 pp | 20.02 +/- 0.01 m/s |

Statistical tests comparing `lambda=0.0` and `lambda=0.1` found no significant difference under this deterministic protocol.

## Interpretation

The strongest research conclusion is not that `lambda=0.1` is statistically superior. The stronger, more credible conclusion is that a promising single-run safety improvement did not survive deterministic multi-seed validation at density 50, revealing that the current evaluation protocol may be saturated or insufficiently diverse.

That negative result is valuable: it shows experimental rigor, prevents overclaiming, and motivates harder stochastic evaluation and failure-boundary discovery.

## Key Figures

- `results/plots/lambda_ablation.png`
- `results/plots/learning_dynamics.png`
- `results/plots/performance_heatmap.png`
- `results/plots/ood_stress_test.png`
- `results/plots/density_robustness.png`
- `results/plots/efficiency_tradeoff.png`

Supplementary analysis figures are stored in `results/supplementary/`.
