import pandas as pd
import numpy as np

def _safe_rate(df, count_col, steps_col="steps"):
    if count_col not in df.columns:
        return 0.0
    steps = df[steps_col].replace(0, np.nan)
    return (df[count_col] / steps).fillna(0).mean()

def calculate_metrics(episode_data):
    """
    Calculate summary metrics from a list of episode data dictionaries.
    """
    df = pd.DataFrame(episode_data)
    crashed = df["crashed"] if "crashed" in df.columns else df.get("collision", pd.Series(0, index=df.index))
    tailgating_rate = _safe_rate(df, "tailgating_count")
    unsafe_lane_change_rate = _safe_rate(df, "unsafe_lane_change_count")
    
    summary = {
        "collision_rate": crashed.mean(),
        "success_rate": df["success"].mean() if "success" in df.columns else (1 - crashed.mean()),
        "avg_reward": df["reward"].mean(),
        "avg_original_reward": df["original_reward"].mean() if "original_reward" in df.columns else 0,
        "avg_speed": df["avg_speed"].mean(),
        "avg_survival_time": df["steps"].mean(),
        "tailgating_rate": tailgating_rate,
        "unsafe_lane_change_rate": unsafe_lane_change_rate,
        "overspeed_rate": _safe_rate(df, "overspeed_count"),
        "safety_violation_score": (crashed * 10 + tailgating_rate + unsafe_lane_change_rate).mean()
        if np.isscalar(tailgating_rate) else 0,
    }
    
    return summary, df

def save_metrics(summary, detailed_df, filename):
    detailed_df.to_csv(f"results/{filename}_detailed.csv", index=False)
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(f"results/{filename}_summary.csv", index=False)
    print(f"Metrics saved to results/{filename}_summary.csv")
