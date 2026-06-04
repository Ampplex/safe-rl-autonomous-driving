import argparse
import os

import numpy as np
import pandas as pd


def _metric_series(df, metric):
    aliases = {
        "collision": ["collision", "crashed", "collision_rate"],
        "success": ["success", "success_rate"],
        "tailgating": ["tailgating", "tailgating_rate"],
        "safety_violation_score": ["safety_violation_score"],
    }
    for col in aliases.get(metric, [metric]):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").dropna()
    raise KeyError(f"Could not find metric '{metric}' in columns: {list(df.columns)}")


def _bootstrap_ci(baseline, safe, n_bootstrap=10000, seed=42, alpha=0.05):
    rng = np.random.default_rng(seed)
    baseline = np.asarray(baseline)
    safe = np.asarray(safe)
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        b = rng.choice(baseline, size=len(baseline), replace=True)
        s = rng.choice(safe, size=len(safe), replace=True)
        diffs[i] = s.mean() - b.mean()
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


def _welch_ttest(baseline, safe):
    try:
        from scipy import stats
    except ImportError:
        return np.nan, np.nan

    stat, p_two_sided = stats.ttest_ind(safe, baseline, equal_var=False)
    if np.isnan(p_two_sided):
        return stat, p_two_sided

    # One-sided p-value for the expected direction: Safe PPO < PPO for violations.
    if np.mean(safe) < np.mean(baseline):
        p_one_sided = p_two_sided / 2
    else:
        p_one_sided = 1 - (p_two_sided / 2)
    return stat, p_one_sided


def run_significance_tests(
    baseline_path="results/multi_seed/baseline_detailed.csv",
    safe_path="results/multi_seed/safeppo_detailed.csv",
    metrics=None,
    output_path="results/statistical_significance.csv",
):
    metrics = metrics or ["collision"]
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(baseline_path)
    if not os.path.exists(safe_path):
        raise FileNotFoundError(safe_path)

    baseline_df = pd.read_csv(baseline_path)
    safe_df = pd.read_csv(safe_path)
    result = _compare_frames(baseline_df, safe_df, metrics)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False)
    print(result.to_string(index=False))
    print(f"Saved {output_path}")
    return result


def _compare_frames(baseline_df, safe_df, metrics):
    rows = []
    for metric in metrics:
        baseline = _metric_series(baseline_df, metric)
        safe = _metric_series(safe_df, metric)
        t_stat, p_value = _welch_ttest(baseline, safe)
        ci_low, ci_high = _bootstrap_ci(baseline, safe)
        rows.append({
            "metric": metric,
            "baseline_mean": baseline.mean(),
            "safeppo_mean": safe.mean(),
            "absolute_difference_safe_minus_baseline": safe.mean() - baseline.mean(),
            "relative_improvement": (baseline.mean() - safe.mean()) / baseline.mean()
            if baseline.mean() != 0 else np.nan,
            "welch_t_stat": t_stat,
            "welch_one_sided_p_value": p_value,
            "bootstrap_diff_ci95_low": ci_low,
            "bootstrap_diff_ci95_high": ci_high,
            "significant_at_0.05": bool(p_value < 0.05) if not np.isnan(p_value) else False,
            "baseline_n": len(baseline),
            "safeppo_n": len(safe),
        })
    return pd.DataFrame(rows)


def run_lambda_significance_tests(
    detailed_path="results/lambda_multi_seed_detailed.csv",
    baseline_lambda=0.0,
    safe_lambda=0.1,
    metrics=None,
    output_path="results/statistical_significance.csv",
):
    metrics = metrics or ["collision", "tailgating_rate", "safety_violation_score"]
    if not os.path.exists(detailed_path):
        raise FileNotFoundError(detailed_path)

    df = pd.read_csv(detailed_path)
    baseline_df = df[df["lambda"].astype(float) == float(baseline_lambda)]
    safe_df = df[df["lambda"].astype(float) == float(safe_lambda)]
    if baseline_df.empty:
        raise ValueError(f"No rows found for baseline lambda={baseline_lambda}")
    if safe_df.empty:
        raise ValueError(f"No rows found for safe lambda={safe_lambda}")

    result = _compare_frames(baseline_df, safe_df, metrics)
    result.insert(1, "baseline_lambda", baseline_lambda)
    result.insert(2, "safe_lambda", safe_lambda)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False)
    print(result.to_string(index=False))
    print(f"Saved {output_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, default="results/multi_seed/baseline_detailed.csv")
    parser.add_argument("--safe", type=str, default="results/multi_seed/safeppo_detailed.csv")
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["collision", "tailgating_rate", "safety_violation_score"],
    )
    parser.add_argument("--output", type=str, default="results/statistical_significance.csv")
    parser.add_argument("--lambda_detailed", type=str, default="results/lambda_multi_seed_detailed.csv")
    parser.add_argument("--baseline_lambda", type=float, default=0.0)
    parser.add_argument("--safe_lambda", type=float, default=0.1)
    parser.add_argument("--use_legacy_files", action="store_true")
    args = parser.parse_args()

    if not args.use_legacy_files and os.path.exists(args.lambda_detailed):
        run_lambda_significance_tests(
            args.lambda_detailed,
            args.baseline_lambda,
            args.safe_lambda,
            args.metrics,
            args.output,
        )
    else:
        run_significance_tests(args.baseline, args.safe, args.metrics, args.output)
