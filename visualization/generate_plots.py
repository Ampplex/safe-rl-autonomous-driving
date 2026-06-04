import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = "results"
PLOTS_DIR = "results/plots"
SUPPLEMENTARY_DIR = "results/supplementary"


def _ensure_dirs():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(SUPPLEMENTARY_DIR, exist_ok=True)


def _save(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved {path}")


def _lambda_from_label(label):
    if pd.isna(label):
        return np.nan
    text = str(label)
    if "Baseline" in text:
        return 0.0
    matches = re.findall(r"\d+(?:\.\d+)?", text)
    return float(matches[-1]) if matches else np.nan


def _load_lambda_data():
    multi_seed_path = f"{RESULTS_DIR}/lambda_multi_seed_summary.csv"
    if os.path.exists(multi_seed_path):
        df = pd.read_csv(multi_seed_path).sort_values("lambda")
        return pd.DataFrame({
            "lambda": df["lambda"],
            "collision_rate": df["collision_mean"],
            "collision_ci95": df.get("collision_ci95", 0),
            "success_rate": df["success_mean"],
            "success_ci95": df.get("success_ci95", 0),
            "tailgating_rate": df["tailgating_rate_mean"],
            "tailgating_ci95": df.get("tailgating_rate_ci95", 0),
            "unsafe_lane_change_rate": df.get("unsafe_lane_change_rate_mean", 0),
            "avg_speed": df["speed_mean"],
            "speed_ci95": df.get("speed_ci95", 0),
            "safety_violation_score": df["safety_violation_score_mean"],
            "violation_ci95": df.get("safety_violation_score_ci95", 0),
        })

    lams = {
        0.0: f"{RESULTS_DIR}/baseline_summary.csv",
        0.1: f"{RESULTS_DIR}/safeppo_summary.csv",
        0.5: f"{RESULTS_DIR}/ablation_lam0.5_summary.csv",
        1.0: f"{RESULTS_DIR}/ablation_lam1.0_summary.csv",
        2.0: f"{RESULTS_DIR}/ablation_lam2.0_summary.csv",
    }
    frames = []
    for lam, path in lams.items():
        if os.path.exists(path):
            row = pd.read_csv(path)
            row["lambda"] = lam
            row["collision_ci95"] = 0.0
            row["success_ci95"] = 0.0
            row["tailgating_ci95"] = 0.0
            row["speed_ci95"] = 0.0
            row["violation_ci95"] = 0.0
            if "unsafe_lane_change_rate" not in row.columns:
                row["unsafe_lane_change_rate"] = 0.0
            if "safety_violation_score" not in row.columns:
                row["safety_violation_score"] = (
                    row["collision_rate"] * 10
                    + row["tailgating_rate"]
                    + row["unsafe_lane_change_rate"]
                )
            frames.append(row)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("lambda")


def plot_benchmark_comparison():
    comparison_path = f"{RESULTS_DIR}/comparison.csv"
    if not os.path.exists(comparison_path):
        print("Skipping PPO vs Safe PPO benchmark: comparison.csv not found.")
        return

    df = pd.read_csv(comparison_path)
    if df.empty or "Model" not in df.columns:
        print("Skipping PPO vs Safe PPO benchmark: comparison.csv is missing model rows.")
        return

    model_labels = df["Model"].replace({
        "Baseline PPO": "PPO",
        "Safe PPO (lambda=0.1)": "Safe PPO",
    })
    colors = ["#D1495B", "#2A9D8F"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    safety_metrics = [
        ("collision_rate", "Collision"),
        ("tailgating_rate", "Tailgating"),
    ]
    x = np.arange(len(safety_metrics))
    width = 0.35

    for idx, (_, row) in enumerate(df.iterrows()):
        values = [row[column] * 100 for column, _ in safety_metrics]
        bars = axes[0].bar(
            x + (idx - 0.5) * width,
            values,
            width,
            label=model_labels.iloc[idx],
            color=colors[idx % len(colors)],
            alpha=0.9,
        )
        axes[0].bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)

    axes[0].set_title("Safety Outcomes")
    axes[0].set_ylabel("Rate (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([label for _, label in safety_metrics])
    axes[0].grid(True, axis="y", linestyle=":", alpha=0.6)
    axes[0].legend()

    efficiency_metrics = [
        ("success_rate", "Success (%)", 100),
        ("avg_speed", "Speed (m/s)", 1),
        ("avg_survival_time", "Survival (s)", 1),
    ]
    x = np.arange(len(efficiency_metrics))

    for idx, (_, row) in enumerate(df.iterrows()):
        values = [row[column] * scale for column, _, scale in efficiency_metrics]
        bars = axes[1].bar(
            x + (idx - 0.5) * width,
            values,
            width,
            label=model_labels.iloc[idx],
            color=colors[idx % len(colors)],
            alpha=0.9,
        )
        axes[1].bar_label(bars, fmt="%.1f", padding=3, fontsize=9)

    axes[1].set_title("Efficiency Outcomes")
    axes[1].set_ylabel("Value")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([label for _, label, _ in efficiency_metrics])
    axes[1].grid(True, axis="y", linestyle=":", alpha=0.6)
    axes[1].legend()

    _save(f"{PLOTS_DIR}/ppo_vs_safeppo_benchmark.png")


def plot_lambda_error_bars(lambda_df):
    if lambda_df.empty:
        print("Skipping lambda plots: no lambda summary data found.")
        return

    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.set_xlabel("Safety Lambda")
    ax1.set_ylabel("Collision / Success Rate (%)")
    ax1.errorbar(
        lambda_df["lambda"],
        lambda_df["collision_rate"] * 100,
        yerr=lambda_df["collision_ci95"] * 100,
        marker="o",
        linewidth=2,
        capsize=5,
        color="tab:red",
        label="Collision Rate",
    )
    ax1.errorbar(
        lambda_df["lambda"],
        lambda_df["success_rate"] * 100,
        yerr=lambda_df["success_ci95"] * 100,
        marker="v",
        linewidth=2,
        capsize=5,
        color="tab:green",
        label="Success Rate",
    )
    ax2 = ax1.twinx()
    ax2.set_ylabel("Tailgating Rate (%)", color="tab:blue")
    ax2.errorbar(
        lambda_df["lambda"],
        lambda_df["tailgating_rate"] * 100,
        yerr=lambda_df["tailgating_ci95"] * 100,
        marker="s",
        linewidth=2,
        capsize=5,
        color="tab:blue",
        label="Tailgating Rate",
    )
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, linestyle=":", alpha=0.6)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right")
    plt.title("Mean Safety Metrics vs Penalty Weight")
    _save(f"{PLOTS_DIR}/lambda_ablation.png")

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        lambda_df["lambda"],
        lambda_df["safety_violation_score"],
        yerr=lambda_df["violation_ci95"],
        marker="x",
        linewidth=2,
        capsize=5,
        color="tab:orange",
    )
    plt.xlabel("Safety Lambda")
    plt.ylabel("Violation Score")
    plt.title("Constraint Violation Profile vs Safety Weight")
    plt.grid(True, linestyle=":", alpha=0.6)
    _save(f"{SUPPLEMENTARY_DIR}/constraint_violations.png")


def plot_efficiency_and_pareto(lambda_df):
    if lambda_df.empty:
        return

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        lambda_df["lambda"],
        lambda_df["avg_speed"],
        yerr=lambda_df["speed_ci95"],
        marker="o",
        linewidth=2,
        capsize=5,
        color="tab:cyan",
        label="Average Speed",
    )
    plt.xlabel("Safety Lambda")
    plt.ylabel("Average Speed")
    plt.title("Efficiency Invariance across Safety Weights")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    _save(f"{PLOTS_DIR}/efficiency_tradeoff.png")

    baseline_rows = lambda_df[lambda_df["lambda"] == 0.0]
    if baseline_rows.empty:
        print("Skipping Pareto plot: baseline lambda=0.0 row missing.")
        return
    baseline = baseline_rows.iloc[0]
    baseline_collision = baseline["collision_rate"]
    if baseline_collision == 0:
        lambda_df = lambda_df.copy()
        lambda_df["safety_gain"] = 0.0
    else:
        lambda_df = lambda_df.copy()
        lambda_df["safety_gain"] = (
            (baseline_collision - lambda_df["collision_rate"]) / baseline_collision * 100
        )
    lambda_df["efficiency_retention"] = lambda_df["avg_speed"] / baseline["avg_speed"] * 100

    plt.figure(figsize=(11, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(lambda_df)))
    for color, (_, row) in zip(colors, lambda_df.iterrows()):
        plt.scatter(
            row["efficiency_retention"],
            row["safety_gain"],
            s=180,
            color=color,
            edgecolor="black",
            alpha=0.85,
        )
        plt.text(
            row["efficiency_retention"],
            row["safety_gain"] + 1,
            f"lambda={row['lambda']}",
            ha="center",
            fontsize=9,
        )
    plt.xlabel("Efficiency Retention (%)")
    plt.ylabel("Safety Gain: Collision Reduction (%)")
    plt.title("Normalized Pareto Frontier")
    plt.grid(True, linestyle=":", alpha=0.6)
    _save(f"{SUPPLEMENTARY_DIR}/pareto_frontier_normalized.png")


def plot_multi_seed_comparison():
    lambda_path = f"{RESULTS_DIR}/lambda_multi_seed_detailed.csv"
    if os.path.exists(lambda_path):
        lambda_df = pd.read_csv(lambda_path)
        b_df = lambda_df[lambda_df["lambda"].astype(float) == 0.0]
        s_df = lambda_df[lambda_df["lambda"].astype(float) == 0.1]
        if b_df.empty or s_df.empty:
            print("Skipping multi-seed comparison: lambda 0.0/0.1 rows missing.")
            return
    else:
        baseline_path = f"{RESULTS_DIR}/multi_seed/baseline_detailed.csv"
        safe_path = f"{RESULTS_DIR}/multi_seed/safeppo_detailed.csv"
        if not os.path.exists(baseline_path) or not os.path.exists(safe_path):
            print("Skipping multi-seed comparison: detailed CSVs missing.")
            return
        b_df = pd.read_csv(baseline_path)
        s_df = pd.read_csv(safe_path)

    metrics = ["collision", "tailgating_rate", "speed"]
    labels = ["Collision %", "Tailgating %", "Speed"]

    def stats(df, metric):
        if metric not in df.columns:
            return 0.0, 0.0
        values = pd.to_numeric(df[metric], errors="coerce").dropna()
        if metric != "speed":
            values = values * 100
        return values.mean(), values.std()

    b_vals = [stats(b_df, metric) for metric in metrics]
    s_vals = [stats(s_df, metric) for metric in metrics]

    title = "PPO vs Safe PPO: Multi-Seed Mean +/- Std"
    stats_path = f"{RESULTS_DIR}/statistical_significance.csv"
    if os.path.exists(stats_path):
        sig = pd.read_csv(stats_path).iloc[0]
        title += f" (collision p={sig['welch_one_sided_p_value']:.3g})"

    plt.figure(figsize=(12, 7))
    x = np.arange(len(metrics))
    width = 0.35
    plt.bar(
        x,
        [v[0] for v in b_vals],
        width,
        yerr=[v[1] for v in b_vals],
        capsize=5,
        label="PPO",
        color="tab:red",
        alpha=0.75,
    )
    plt.bar(
        x + width,
        [v[0] for v in s_vals],
        width,
        yerr=[v[1] for v in s_vals],
        capsize=5,
        label="Safe PPO",
        color="tab:blue",
        alpha=0.75,
    )
    plt.ylabel("Value")
    plt.title(title)
    plt.xticks(x + width / 2, labels)
    plt.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.legend()
    _save(f"{SUPPLEMENTARY_DIR}/multi_seed_comparison.png")


def _load_training_rollout_data(log_root="logs/ppo"):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("Skipping training rollout plot: TensorBoard is not installed.")
        return pd.DataFrame()

    records = []
    for event_path in sorted(Path(log_root).glob("PPO_*/*tfevents*")):
        accumulator = EventAccumulator(str(event_path))
        accumulator.Reload()
        scalar_tags = set(accumulator.Tags().get("scalars", []))
        required_tags = {"rollout/ep_rew_mean", "rollout/ep_len_mean"}
        if not required_tags.issubset(scalar_tags):
            continue

        reward_events = {event.step: event.value for event in accumulator.Scalars("rollout/ep_rew_mean")}
        length_events = {event.step: event.value for event in accumulator.Scalars("rollout/ep_len_mean")}
        for step in sorted(set(reward_events) & set(length_events)):
            records.append({
                "source_run": event_path.parent.name,
                "step": step,
                "episode_reward_mean": reward_events[step],
                "episode_length_mean": length_events[step],
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    max_steps = df.groupby("source_run")["step"].max().sort_values()
    source_run = max_steps.index[-1]
    return df[df["source_run"] == source_run].sort_values("step")


def plot_training_rollout_progress():
    rollout = _load_training_rollout_data()
    if rollout.empty:
        print("Skipping training rollout progress: no rollout TensorBoard data found.")
        return

    rollout.to_csv(f"{RESULTS_DIR}/training_rollout_progress.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(
        rollout["step"],
        rollout["episode_reward_mean"],
        marker="o",
        linewidth=2.5,
        color="tab:blue",
    )
    ax1.set_title("Training Rollout Reward")
    ax1.set_xlabel("Training Timesteps")
    ax1.set_ylabel("Mean Episode Reward")
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.plot(
        rollout["step"],
        rollout["episode_length_mean"],
        marker="o",
        linewidth=2.5,
        color="tab:green",
    )
    ax2.set_title("Training Rollout Episode Length")
    ax2.set_xlabel("Training Timesteps")
    ax2.set_ylabel("Mean Episode Length")
    ax2.grid(True, linestyle=":", alpha=0.6)

    _save(f"{PLOTS_DIR}/training_rollout_progress.png")


def _load_density_data():
    frames = []
    for path in [f"{RESULTS_DIR}/traffic_density_results.csv", f"{RESULTS_DIR}/ood_results.csv"]:
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=["density"], keep="last").sort_values("density")


def plot_density_robustness():
    df = _load_density_data()
    if df.empty:
        print("Skipping density robustness: no density CSVs found.")
        return

    plt.figure(figsize=(12, 7))
    plt.plot(df["density"], df["collision_rate"] * 100, marker="o", label="Collision Rate")
    plt.plot(df["density"], df["tailgating_rate"] * 100, marker="s", label="Tailgating Rate")
    plt.plot(df["density"], df["success_rate"] * 100, marker="v", label="Success Rate")
    plt.axvline(x=30, color="gray", linestyle="--", label="Training Density")
    plt.xlabel("Traffic Density (Vehicles)")
    plt.ylabel("Rate (%)")
    plt.title("Density Robustness with Success Rate")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    _save(f"{PLOTS_DIR}/density_robustness.png")

    plt.figure(figsize=(12, 7))
    plt.plot(df["density"], df["success_rate"] * 100, marker="o", linewidth=3, color="tab:purple")
    collapsed = df[df["success_rate"] < 0.8]
    if not collapsed.empty:
        threshold = collapsed.iloc[0]["density"]
        plt.axvline(threshold, color="tab:red", linestyle="--", label=f"Failure threshold ~= {threshold:g}")
    else:
        plt.text(
            df["density"].max(),
            df["success_rate"].min() * 100,
            "No <80% collapse observed",
            ha="right",
            va="bottom",
        )
    plt.axvline(x=30, color="gray", linestyle="--", label="Training Density")
    plt.xlabel("Traffic Density (Vehicles)")
    plt.ylabel("Success Rate (%)")
    plt.title("Failure Boundary Discovery")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    _save(f"{PLOTS_DIR}/ood_stress_test.png")


def plot_performance_heatmap():
    df = _load_density_data()
    if df.empty:
        return
    heatmap = df.set_index("density")[["collision_rate", "tailgating_rate", "success_rate", "avg_speed"]]
    heatmap = heatmap.copy()
    for col in ["collision_rate", "tailgating_rate", "success_rate"]:
        heatmap[col] *= 100

    data = heatmap.values
    rows = heatmap.index
    cols = ["Collision %", "Tailgating %", "Success %", "Avg Speed"]

    fig, ax = plt.subplots(figsize=(12, max(6, len(rows) * 0.7)))
    im = ax.imshow(data, cmap="YlGnBu")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels(cols)
    ax.set_yticklabels(rows)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", color="black")
    ax.set_title("Performance Heatmap across Traffic Densities")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(f"{PLOTS_DIR}/performance_heatmap.png")


def _policy_scores(path, fallback_summary=False):
    df = pd.read_csv(path)
    if fallback_summary:
        row = df.mean(numeric_only=True)
        collision = row.get("collision_rate", row.get("collision", 0.0))
        tailgating = row.get("tailgating_rate", row.get("tailgating", 0.0))
        success = row.get("success_rate", row.get("success", 1 - collision))
        speed = row.get("avg_speed", row.get("speed", 20.0))
        survival = row.get("avg_survival_time", row.get("survival_time", 30.0))
    else:
        collision = df.get("collision", pd.Series(0)).mean()
        tailgating = df.get("tailgating_rate", df.get("tailgating", pd.Series(0))).mean()
        success = df.get("success", pd.Series(1 - collision, index=df.index)).mean()
        speed = df.get("speed", pd.Series(20.0, index=df.index)).mean()
        survival = df.get("survival_time", pd.Series(30.0, index=df.index)).mean()

    return [
        success,
        1 - collision,
        1 - min(1.0, tailgating),
        min(1.2, speed / 25.0),
        min(1.2, survival / 30.0),
    ]


def _policy_scores_from_df(df):
    collision = df.get("collision", pd.Series(0, index=df.index)).mean()
    tailgating = df.get("tailgating_rate", df.get("tailgating", pd.Series(0, index=df.index))).mean()
    success = df.get("success", pd.Series(1 - collision, index=df.index)).mean()
    speed = df.get("speed", pd.Series(20.0, index=df.index)).mean()
    survival = df.get("survival_time", pd.Series(30.0, index=df.index)).mean()

    return [
        success,
        1 - collision,
        1 - min(1.0, tailgating),
        min(1.2, speed / 25.0),
        min(1.2, survival / 30.0),
    ]


def plot_radar_chart():
    labels = ["Success", "Collision Avoid.", "Tailgating Avoid.", "Efficiency", "Survival"]
    lambda_path = f"{RESULTS_DIR}/lambda_multi_seed_detailed.csv"
    if os.path.exists(lambda_path):
        lambda_df = pd.read_csv(lambda_path)
        b_df = lambda_df[lambda_df["lambda"].astype(float) == 0.0]
        s_df = lambda_df[lambda_df["lambda"].astype(float) == 0.1]
        if b_df.empty or s_df.empty:
            print("Skipping radar chart: lambda 0.0/0.1 rows missing.")
            return
        baseline = _policy_scores_from_df(b_df)
        safe = _policy_scores_from_df(s_df)
    else:
        baseline_path = f"{RESULTS_DIR}/multi_seed/baseline_detailed.csv"
        safe_path = f"{RESULTS_DIR}/multi_seed/safeppo_detailed.csv"
        fallback = False
        if not os.path.exists(baseline_path) or not os.path.exists(safe_path):
            baseline_path = f"{RESULTS_DIR}/baseline_summary.csv"
            safe_path = f"{RESULTS_DIR}/safeppo_summary.csv"
            fallback = True
        if not os.path.exists(baseline_path) or not os.path.exists(safe_path):
            print("Skipping radar chart: baseline/safe summaries missing.")
            return
        baseline = _policy_scores(baseline_path, fallback_summary=fallback)
        safe = _policy_scores(safe_path, fallback_summary=fallback)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    baseline += baseline[:1]
    safe += safe[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    ax.fill(angles, baseline, color="tab:red", alpha=0.25, label="PPO")
    ax.plot(angles, baseline, color="tab:red", linewidth=2)
    ax.fill(angles, safe, color="tab:blue", alpha=0.25, label="Safe PPO")
    ax.plot(angles, safe, color="tab:blue", linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title("PPO vs Safe PPO Performance Fingerprint", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    _save(f"{SUPPLEMENTARY_DIR}/radar_comparison.png")


def plot_behavior_analysis():
    path = f"{RESULTS_DIR}/lambda_behavior_analysis.csv"
    if not os.path.exists(path):
        print("Skipping behavior analysis plot: lambda_behavior_analysis.csv missing.")
        return
    df = pd.read_csv(path)
    if "lambda" not in df.columns:
        df["lambda"] = df["model"].apply(_lambda_from_label)
    df = df.dropna(subset=["lambda"]).sort_values("lambda")

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.plot(df["lambda"], df["lane_changes"], marker="o", linewidth=2, color="tab:purple", label="Lane Changes")
    ax1.set_xlabel("Safety Lambda")
    ax1.set_ylabel("Lane Changes per Episode")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax2 = ax1.twinx()
    if "collision" in df.columns:
        ax2.plot(df["lambda"], df["collision"] * 100, marker="s", color="tab:red", label="Collision Rate")
        ax2.set_ylabel("Collision Rate (%)")
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
    plt.title("Behavior Analysis: Lane Changes vs Safety Weight")
    _save(f"{SUPPLEMENTARY_DIR}/lane_changes_vs_lambda.png")


def plot_reward_breakdown():
    path = f"{RESULTS_DIR}/reward_component_breakdown.csv"
    if not os.path.exists(path):
        print("Skipping reward breakdown: run evaluation/reward_breakdown_eval.py first.")
        return
    df = pd.read_csv(path).sort_values("lambda")
    labels = [f"{row['lambda']}" for _, row in df.iterrows()]
    x = np.arange(len(df))
    width = 0.55

    plt.figure(figsize=(12, 7))
    plt.bar(x, df["progress_reward"], width, label="Progress Reward", color="tab:green", alpha=0.8)
    bottom = np.zeros(len(df))
    penalties = [
        ("collision_penalty", "Collision Penalty", "tab:red"),
        ("tailgating_penalty", "Tailgating Penalty", "tab:orange"),
        ("lane_change_penalty", "Lane Change Penalty", "tab:purple"),
        ("speed_penalty", "Speed Penalty", "tab:blue"),
    ]
    for col, label, color in penalties:
        values = -df[col]
        plt.bar(x, values, width, bottom=bottom, label=label, color=color, alpha=0.75)
        bottom += values
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(x, labels)
    plt.xlabel("Safety Lambda")
    plt.ylabel("Episode Reward Component")
    plt.title("Reward Component Breakdown by Safety Weight")
    plt.legend()
    plt.grid(True, axis="y", linestyle=":", alpha=0.6)
    _save(f"{SUPPLEMENTARY_DIR}/reward_component_breakdown.png")


def plot_sensitivity_surface():
    path = f"{RESULTS_DIR}/sensitivity_surface.csv"
    if not os.path.exists(path):
        print("Skipping sensitivity surface: run experiments/exp7_sensitivity_surface.py first.")
        return
    df = pd.read_csv(path)
    if len(df) < 3:
        print("Skipping sensitivity surface: need at least 3 evaluated grid points.")
        return

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    surface_metric = "safety_score" if "safety_score" in df.columns else "success"
    ax.plot_trisurf(
        df["collision_penalty"],
        df["tailgating_penalty"],
        df[surface_metric],
        cmap="viridis",
        edgecolor="none",
        alpha=0.9,
    )
    ax.set_xlabel("Collision Penalty")
    ax.set_ylabel("Tailgating Penalty")
    ax.set_zlabel(surface_metric.replace("_", " ").title())
    ax.set_title("Sensitivity Surface for Reward Penalties")
    _save(f"{SUPPLEMENTARY_DIR}/sensitivity_surface.png")


def generate_plots():
    _ensure_dirs()
    lambda_df = _load_lambda_data()
    plot_benchmark_comparison()
    plot_lambda_error_bars(lambda_df)
    plot_efficiency_and_pareto(lambda_df)
    plot_multi_seed_comparison()
    plot_training_rollout_progress()
    plot_density_robustness()
    plot_performance_heatmap()
    plot_radar_chart()
    plot_behavior_analysis()
    plot_reward_breakdown()
    plot_sensitivity_surface()


if __name__ == "__main__":
    generate_plots()
