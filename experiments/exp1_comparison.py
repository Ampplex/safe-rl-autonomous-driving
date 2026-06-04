import pandas as pd

def create_comparison():
    baseline = pd.read_csv("results/baseline_summary.csv")
    safeppo = pd.read_csv("results/safeppo_summary.csv")
    
    baseline['Model'] = 'Baseline PPO'
    safeppo['Model'] = 'Safe PPO (lambda=0.1)'
    
    comparison = pd.concat([baseline, safeppo], ignore_index=True)
    
    # Reorder columns to put Model first
    cols = ['Model'] + [c for c in comparison.columns if c != 'Model']
    comparison = comparison[cols]
    
    comparison.to_csv("results/comparison.csv", index=False)
    
    print("\n" + "="*40)
    print("      Model Comparison (Experiment 1)")
    print("="*40)
    print(comparison[['Model', 'collision_rate', 'tailgating_rate', 'avg_speed', 'success_rate']].to_string(index=False))
    print("="*40)
    print("Results saved to results/comparison.csv")

if __name__ == "__main__":
    try:
        create_comparison()
    except FileNotFoundError:
        print("Error: baseline_summary.csv or safeppo_summary.csv not found.")
        print("Please run baseline evaluation and Safe PPO evaluation first.")
