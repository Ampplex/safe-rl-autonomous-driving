# 🚗 Safe-RL Autonomous Driving: Final Research Strategy & Execution Manifest

## 📌 Mission Statement
Elevate the project from a technical MVP to a rigorous scientific study for the **Amazon ML Summer School 2026**. The focus is on empirical validation, statistical significance, and deep behavioral analysis of safety-constrained Reinforcement Learning.

---

## 🛠 Tier 1: Statistical Rigor & Convergence (Highest ROI)

### 1.1 Multiple Seeds & Error Bars
- **Goal:** Test whether the single-run Safe PPO collision reduction survives repeated seeded evaluation.
- **Execution:** Evaluated λ ∈ `{0.0, 0.1, 0.5, 1.0, 2.0}` across 5 independent seed groups (`42, 123, 456, 789, 2025`) with 20 episodes per seed at density 50.
- **Metric:** Report results as mean with 95% CI.
- **Status:** [COMPLETED ✅]
- **Finding:** Deterministic evaluation produced identical aggregate behavior across all λ values: collision rate `2.0% ± 2.8 pp`, success rate `98.0% ± 2.8 pp`, tailgating rate `13.6% ± 5.6 pp`, and average speed `20.02 ± 0.01 m/s`.

### 1.2 Learning Dynamics Analysis
- **Goal:** Visualize the "Price of Safety" in terms of training speed and convergence stability.
- **Visuals:** `Reward vs. Steps` and `Collision Rate vs. Steps` comparison.
- **Insight:** Show the training-dynamics tradeoff; final policy superiority must be judged by the multi-seed benchmark.
- **Status:** [DATA GENERATED ✅]

### 1.3 Statistical Significance
- **Goal:** Quantify confidence in the improvement.
- **Execution:** Application of statistical tests (T-test/Bootstrap) between baseline and Safe PPO.
- **Status:** [COMPLETED ✅]
- **Finding:** No statistically significant difference was detected between λ=0.0 and λ=0.1 in the completed deterministic multi-seed benchmark (`p=0.5` for collision, tailgating rate, and safety violation score). This invalidates a strong `p < 0.05` claim for the current density-50 deterministic protocol.

---

## 🔬 Tier 2: Behavioral & Constraint Analysis

### 2.1 Performance Heatmap
- **Goal:** Identify "phase transitions" in policy performance across traffic regimes.
- **Metrics:** Collision, Success, Tailgating, and Speed across densities [20, 50, 100, 150, 200, 250, 300].
- **Status:** [GENERATED ✅]

### 2.2 Radar Charts (Policy Fingerprinting)
- **Goal:** Provide a high-level visual "fingerprint" of the trade-offs.
- **Metrics:** Success Rate, Collision Avoidance, Tailgating Avoidance, Efficiency, Survival Time.
- **Status:** [GENERATED ✅]

### 2.3 Constraint Violation Analysis
- **Goal:** Track the "Total Safety Cost" vs. Lambda.
- **Violation Score:** $10 \times Collisions + Tailgating + Unsafe Lane Changes$.
- **Status:** [GENERATED ✅]
- **Finding:** Violation scores were identical across deterministic λ policies in the multi-seed evaluation; the reward component breakdown still shows the expected scaling of tailgating penalties with λ.

---

## 📡 Tier 3: Generalization & Boundary Discovery

### 3.1 Failure Boundary Discovery (The "Stress Limit")
- **Goal:** Find the exact density where the policy collapses.
- **Range:** Extending OOD tests from 300 up to 500 vehicles.
- **Observation:** Determine if the policy fails gracefully or catastrophically.
- **Status:** [RUNNING 🔄] Density 350+ extension is still active.

### 3.2 Lane Change & Aggression Analysis
- **Goal:** Explain *how* the agent achieves safety.
- **Metric:** Lane Change Frequency vs. λ.
- **Hypothesis:** Higher λ leads to more "patient" driving (fewer lane changes).
- **Status:** [COMPLETED ✅]
- **Finding:** Deterministic policies made zero lane-change actions in the behavior analysis, so no λ-dependent lane-change effect was observed.

---

## 📈 Final Visualization Suite
The project will deliver the following research-grade figures:
1. `pareto_frontier_normalized.png` - Objective dominance.
2. `multi_seed_comparison.png` - Multi-seed error-bar comparison.
3. `performance_heatmap.png` - Comprehensive density response.
4. `learning_dynamics.png` - Training convergence profiles.
5. `ood_stress_test.png` - Failure boundary identification.
6. `radar_comparison.png` - Multi-objective policy comparison.
7. `constraint_violations.png` - Safety violation score vs λ.
8. `lane_changes_vs_lambda.png` - Behavior analysis vs λ.
9. `reward_component_breakdown.png` - Reward accounting by λ.

---

## 📅 Execution Status (Live)
- **Multi-Seed Benchmark:** [COMPLETE ✅] 500/500 episodes saved to `results/lambda_multi_seed_detailed.csv`.
- **Statistical Significance:** [COMPLETE ✅] Saved to `results/statistical_significance.csv`; no significant λ=0.1 improvement under the deterministic density-50 protocol.
- **Behavioral Eval:** [COMPLETE ✅] Saved to `results/lambda_behavior_analysis.csv`.
- **Reward Breakdown:** [COMPLETE ✅] Saved to `results/reward_component_breakdown.csv`.
- **Plot Refresh:** [COMPLETE ✅] Updated figures in `results/plots/`.
- **OOD Extension:** [RUNNING 🔄] PID `62617` - Testing densities 350-500; density 350 has reached at least episode 10/20.

**Next Step:** Let the OOD extension continue writing density checkpoints, then rerun `venv/bin/python visualization/generate_plots.py` after new OOD rows are saved.
