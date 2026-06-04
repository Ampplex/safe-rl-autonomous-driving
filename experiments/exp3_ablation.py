import subprocess
import pandas as pd
import os
import sys

def run_experiment(safety_lambda):
    print(f"\n{'='*20}")
    print(f"Running Ablation for Lambda = {safety_lambda}")
    print(f"{'='*20}\n")
    
    python_exe = sys.executable
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + ":" + os.getcwd()
    
    # 1. Train
    train_cmd = [
        python_exe, "training/train_safeppo.py",
        "--safety_lambda", str(safety_lambda)
    ]
    subprocess.run(train_cmd, env=env, check=True)
    
    # 2. Evaluate
    model_path = f"models/safeppo_agent_lam{safety_lambda}.zip"
    eval_name = f"ablation_lam{safety_lambda}"
    eval_cmd = [
        python_exe, "evaluation/evaluate.py",
        "--model", model_path,
        "--name", eval_name,
        "--safety_lambda", str(safety_lambda),
        "--episodes", "100"
    ]
    subprocess.run(eval_cmd, env=env, check=True)
    
    # Return path to the summary csv
    return f"results/{eval_name}_summary.csv"

def main():
    lambdas = [0.5, 1.0, 2.0]
    results = []
    
    # Create results dir if not exists
    os.makedirs("results", exist_ok=True)
    
    for lam in lambdas:
        summary_path = run_experiment(lam)
        df = pd.read_csv(summary_path)
        df['lambda'] = lam
        results.append(df)
        
    # Combine all ablation results
    if results:
        final_df = pd.concat(results, ignore_index=True)
        final_df.to_csv("results/lambda_ablation_new.csv", index=False)
        print("\nAblation Study Complete. Results saved to results/lambda_ablation_new.csv")

if __name__ == "__main__":
    main()
