"""
Real-time visualization of training progress from saved metrics.
Useful for monitoring long-running training sessions.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import time

def load_metrics(results_dir):
    """Load metrics from JSON file."""
    metrics_path = os.path.join(results_dir, 'metrics.json')
    if not os.path.exists(metrics_path):
        return None
    
    with open(metrics_path, 'r') as f:
        return json.load(f)

def plot_metrics(metrics):
    """Create comprehensive visualization of metrics."""
    variant_names = metrics['variants']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Primal Hunt DQN Training - Progress Monitor', 
                 fontsize=16, fontweight='bold')
    
    # 1. Episode Rewards
    ax = axes[0, 0]
    for idx, name in enumerate(variant_names):
        if len(metrics['episode_rewards'][idx]) > 0:
            episodes = range(len(metrics['episode_rewards'][idx]))
            rewards = metrics['episode_rewards'][idx]
            ax.plot(episodes, rewards, label=name, color=colors[idx], linewidth=2, alpha=0.7)
            
            # Add moving average
            if len(rewards) >= 20:
                window = 20
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(rewards)), moving_avg, 
                       color=colors[idx], linewidth=3, alpha=1.0)
    
    ax.set_title('Episode Rewards (with 20-episode moving average)', fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 2. Episode Length
    ax = axes[0, 1]
    for idx, name in enumerate(variant_names):
        if len(metrics['episode_lengths'][idx]) > 0:
            episodes = range(len(metrics['episode_lengths'][idx]))
            ax.plot(episodes, metrics['episode_lengths'][idx], 
                   label=name, color=colors[idx], linewidth=2, alpha=0.7)
    
    ax.set_title('Episode Length', fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 3. Food Collected
    ax = axes[0, 2]
    for idx, name in enumerate(variant_names):
        if len(metrics['episode_food'][idx]) > 0:
            episodes = range(len(metrics['episode_food'][idx]))
            ax.plot(episodes, metrics['episode_food'][idx], 
                   label=name, color=colors[idx], linewidth=2, alpha=0.7)
    
    ax.axhline(y=10, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Success Threshold')
    ax.set_title('Food Collected per Episode', fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Food Amount')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 4. Success Rate
    ax = axes[1, 0]
    for idx, name in enumerate(variant_names):
        if len(metrics['success_rates'][idx]) > 0:
            episodes = range(len(metrics['success_rates'][idx]))
            ax.plot(episodes, metrics['success_rates'][idx], 
                   label=name, color=colors[idx], linewidth=2)
    
    ax.set_title('Success Rate (100-episode window)', fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate')
    ax.set_ylim([0, 1])
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 5. Training Progress Bar
    ax = axes[1, 1]
    ax.axis('off')
    
    max_episodes = max([len(r) for r in metrics['episode_rewards']])
    
    y_pos = 0.8
    for idx, name in enumerate(variant_names):
        episodes_done = len(metrics['episode_rewards'][idx])
        progress = episodes_done / 500.0  # Assuming 500 total episodes
        
        # Progress bar
        ax.barh(y_pos, progress, height=0.15, color=colors[idx], alpha=0.7)
        ax.barh(y_pos, 1.0, height=0.15, color='lightgray', alpha=0.3)
        
        # Text
        ax.text(0.5, y_pos, f"{name}: {episodes_done}/500 ({progress*100:.1f}%)", 
               ha='center', va='center', fontweight='bold', fontsize=10)
        
        y_pos -= 0.25
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title('Training Progress', fontweight='bold')
    
    # 6. Performance Summary
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = "Latest Performance (last 10 episodes):\n\n"
    
    for idx, name in enumerate(variant_names):
        if len(metrics['episode_rewards'][idx]) >= 10:
            recent_reward = np.mean(metrics['episode_rewards'][idx][-10:])
            recent_food = np.mean(metrics['episode_food'][idx][-10:])
            recent_success = metrics['success_rates'][idx][-1] if metrics['success_rates'][idx] else 0
            
            summary_text += f"{name}:\n"
            summary_text += f"  Reward: {recent_reward:.1f}\n"
            summary_text += f"  Food: {recent_food:.1f}\n"
            summary_text += f"  Success: {recent_success:.1%}\n\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig

def monitor_training(results_dir, refresh_interval=10):
    """Monitor training progress in real-time."""
    print(f"Monitoring training in: {results_dir}")
    print(f"Refresh interval: {refresh_interval} seconds")
    print("Press Ctrl+C to stop\n")
    
    plt.ion()
    
    try:
        while True:
            metrics = load_metrics(results_dir)
            
            if metrics is None:
                print("Waiting for metrics file...")
                time.sleep(refresh_interval)
                continue
            
            # Check if training is complete
            max_episodes = max([len(r) for r in metrics['episode_rewards']])
            
            print(f"\rProgress: {max_episodes}/500 episodes", end='', flush=True)
            
            # Plot
            fig = plot_metrics(metrics)
            plt.pause(0.1)
            
            if max_episodes >= 500:
                print("\n\nTraining complete!")
                plt.ioff()
                plt.show()
                break
            
            plt.close(fig)
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        plt.ioff()
        plt.show()

def plot_final_results(results_dir):
    """Plot final results after training is complete."""
    metrics = load_metrics(results_dir)
    
    if metrics is None:
        print(f"Error: No metrics found in {results_dir}")
        return
    
    fig = plot_metrics(metrics)
    
    # Save figure
    output_path = os.path.join(results_dir, 'progress_visualization.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Monitor live training: python visualize_progress.py <results_dir> [refresh_interval]")
        print("  Plot final results:    python visualize_progress.py <results_dir> --final")
        print("\nExample:")
        print("  python visualize_progress.py results_20241014_120000")
        print("  python visualize_progress.py results_20241014_120000 --final")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    if not os.path.exists(results_dir):
        print(f"Error: Directory '{results_dir}' not found")
        sys.exit(1)
    
    if len(sys.argv) > 2 and sys.argv[2] == '--final':
        plot_final_results(results_dir)
    else:
        refresh_interval = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        monitor_training(results_dir, refresh_interval)
