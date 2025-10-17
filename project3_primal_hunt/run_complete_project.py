"""
Complete Project 3 Runner
Runs demo, training, evaluation, and generates final report.
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def run_command(cmd, description):
    """Run command and handle errors."""
    print(f"Running: {description}")
    print(f"Command: {cmd}\n")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=False, text=True)
        print(f"✓ {description} completed successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with error code {e.returncode}\n")
        return False

def main():
    """Run complete project pipeline."""
    start_time = time.time()
    
    print_section("PROJECT 3: PRIMAL HUNT - COMPLETE PIPELINE")
    
    print("This script will:")
    print("1. Run environment demo")
    print("2. Train all 4 DQN variants (500 episodes each)")
    print("3. Evaluate trained agents (100 episodes each)")
    print("4. Generate comprehensive visualizations")
    print("5. Create statistical analysis report")
    print("\nEstimated time: 30-60 minutes\n")
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Step 1: Demo
    print_section("STEP 1: Environment Demo")
    if not run_command("python demo.py", "Environment Demo"):
        print("Warning: Demo failed, but continuing with training...")
    
    # Step 2: Training
    print_section("STEP 2: Training DQN Variants")
    print("Training 4 variants:")
    print("  - Full DQN (with replay + target network)")
    print("  - No Replay (target network only)")
    print("  - No Target Network (replay only)")
    print("  - Vanilla Q-Learning (neither)")
    print()
    
    if not run_command("python train.py", "DQN Training"):
        print("Error: Training failed. Cannot continue.")
        return
    
    # Find results directory
    results_dirs = [d for d in os.listdir('.') if d.startswith('results_')]
    if not results_dirs:
        print("Error: No results directory found.")
        return
    
    latest_results = sorted(results_dirs)[-1]
    print(f"\nResults saved to: {latest_results}")
    
    # Step 3: Evaluation
    print_section("STEP 3: Evaluating Trained Agents")
    if not run_command(f"python evaluate.py {latest_results}", "Agent Evaluation"):
        print("Error: Evaluation failed.")
        return
    
    # Step 4: Summary
    print_section("PROJECT COMPLETE!")
    
    elapsed_time = time.time() - start_time
    print(f"Total time: {elapsed_time/60:.1f} minutes\n")
    
    print("Generated Files:")
    print(f"  📁 {latest_results}/")
    print(f"     ├── agent_Full_DQN.pt")
    print(f"     ├── agent_No_Replay.pt")
    print(f"     ├── agent_No_Target_Network.pt")
    print(f"     ├── agent_Vanilla_Q-Learning.pt")
    print(f"     ├── metrics.json")
    print(f"     ├── training_progress.png")
    print(f"     └── evaluation/")
    print(f"         ├── Full_DQN_episode.gif")
    print(f"         ├── No_Replay_episode.gif")
    print(f"         ├── No_Target_Network_episode.gif")
    print(f"         ├── Vanilla_Q-Learning_episode.gif")
    print(f"         ├── evaluation_comparison.png")
    print(f"         ├── evaluation_results.json")
    print(f"         └── statistical_analysis.txt")
    
    print("\n📊 View Results:")
    print(f"  - Training progress: {latest_results}/training_progress.png")
    print(f"  - Evaluation plots: {latest_results}/evaluation/evaluation_comparison.png")
    print(f"  - Statistical analysis: {latest_results}/evaluation/statistical_analysis.txt")
    print(f"  - Episode animations: {latest_results}/evaluation/*.gif")
    
    print("\n📝 Next Steps:")
    print("  1. Review the statistical analysis report")
    print("  2. Examine the training and evaluation plots")
    print("  3. Watch the episode animations")
    print("  4. Fill in the LaTeX report with actual results")
    print("  5. Compile report: pdflatex report_template.tex")
    
    print("\n" + "="*70)
    print("  Thank you for using Project 3: Primal Hunt!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
