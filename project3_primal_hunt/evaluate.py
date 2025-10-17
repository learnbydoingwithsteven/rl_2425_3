"""
Evaluation script for trained DQN agents.
Generates comprehensive visualizations and performance metrics.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import json
import os
from tqdm import tqdm

from environment import PrimalHuntEnv
from dqn_agent import DQNAgent


class EpisodeVisualizer:
    """Visualize agent episodes with environment rendering."""
    
    def __init__(self, env, agent, variant_name):
        self.env = env
        self.agent = agent
        self.variant_name = variant_name
        self.frames = []
    
    def run_episode(self, render=True):
        """Run single episode and optionally record frames."""
        state = self.env.reset()
        done = False
        episode_reward = 0
        
        if render:
            self.frames = []
            self.frames.append(self.env.render())
        
        while not done:
            action = self.agent.select_action(state, training=False)
            next_state, reward, done, info = self.env.step(action)
            
            if render:
                self.frames.append(self.env.render())
            
            episode_reward += reward
            state = next_state
        
        return episode_reward, self.env.food_collected, info
    
    def create_animation(self, filepath, fps=5):
        """Create animated GIF of episode."""
        if len(self.frames) == 0:
            print("No frames to animate. Run episode with render=True first.")
            return
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        def update(frame_idx):
            ax.clear()
            ax.imshow(self.frames[frame_idx])
            ax.set_title(f'{self.variant_name} - Step {frame_idx}/{len(self.frames)-1}')
            ax.axis('off')
            
            # Add legend
            legend_elements = [
                patches.Patch(facecolor='blue', label='Hunter'),
                patches.Patch(facecolor='green', label='Food'),
                patches.Patch(facecolor='yellow', label='Friendly Animal'),
                patches.Patch(facecolor='orange', label='Hostile Animal'),
                patches.Patch(facecolor='red', label='Hostile Tribe'),
                patches.Patch(facecolor='gray', label='Obstacle')
            ]
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        anim = FuncAnimation(fig, update, frames=len(self.frames), interval=1000/fps)
        anim.save(filepath, writer=PillowWriter(fps=fps))
        plt.close()
        print(f"Animation saved to {filepath}")


def evaluate_agents(results_dir, num_eval_episodes=100):
    """Evaluate all trained agents and generate comprehensive results."""
    
    # Load metrics
    with open(os.path.join(results_dir, 'metrics.json'), 'r') as f:
        training_metrics = json.load(f)
    
    variant_names = training_metrics['variants']
    
    # Create evaluation directory
    eval_dir = os.path.join(results_dir, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    
    # Evaluation results
    eval_results = {
        'variants': variant_names,
        'eval_rewards': [],
        'eval_food': [],
        'eval_success_rates': [],
        'eval_episode_lengths': []
    }
    
    print("\n" + "="*60)
    print("Evaluating Trained Agents")
    print("="*60 + "\n")
    
    for idx, variant_name in enumerate(variant_names):
        print(f"\nEvaluating: {variant_name}")
        
        # Load agent
        agent_path = os.path.join(results_dir, f"agent_{variant_name.replace(' ', '_')}.pt")
        
        # Determine configuration
        use_replay = 'No Replay' not in variant_name and 'Vanilla' not in variant_name
        use_target = 'No Target' not in variant_name and 'Vanilla' not in variant_name
        
        # Create environment and agent
        env = PrimalHuntEnv(grid_size=15, max_steps=200)
        agent = DQNAgent(
            state_dim=env.state_dim,
            action_dim=env.action_space,
            use_replay=use_replay,
            use_target_network=use_target,
            device='cpu'
        )
        agent.load(agent_path)
        agent.epsilon = 0.0  # Greedy policy for evaluation
        
        # Run evaluation episodes
        rewards = []
        food_collected = []
        success_count = 0
        episode_lengths = []
        
        for ep in tqdm(range(num_eval_episodes), desc=f"Evaluating {variant_name}"):
            visualizer = EpisodeVisualizer(env, agent, variant_name)
            reward, food, info = visualizer.run_episode(render=(ep == 0))
            
            rewards.append(reward)
            food_collected.append(food)
            episode_lengths.append(info.get('step', 0))
            
            if info.get('reason') == 'success':
                success_count += 1
            
            # Save animation for first episode
            if ep == 0:
                anim_path = os.path.join(eval_dir, f'{variant_name.replace(" ", "_")}_episode.gif')
                visualizer.create_animation(anim_path)
        
        # Store results
        eval_results['eval_rewards'].append(rewards)
        eval_results['eval_food'].append(food_collected)
        eval_results['eval_success_rates'].append(success_count / num_eval_episodes)
        eval_results['eval_episode_lengths'].append(episode_lengths)
        
        print(f"  Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"  Mean Food: {np.mean(food_collected):.2f} ± {np.std(food_collected):.2f}")
        print(f"  Success Rate: {success_count/num_eval_episodes:.2%}")
    
    # Save evaluation results
    with open(os.path.join(eval_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    # Generate comparison plots
    generate_comparison_plots(eval_results, eval_dir)
    
    # Generate statistical analysis
    generate_statistical_analysis(training_metrics, eval_results, eval_dir)
    
    print(f"\n{'='*60}")
    print(f"Evaluation Complete! Results saved to: {eval_dir}")
    print(f"{'='*60}\n")


def generate_comparison_plots(eval_results, output_dir):
    """Generate comprehensive comparison plots."""
    
    variant_names = eval_results['variants']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('DQN Ablation Study - Evaluation Results', fontsize=16, fontweight='bold')
    
    # 1. Reward Distribution
    ax = axes[0, 0]
    positions = range(len(variant_names))
    bp = ax.boxplot(eval_results['eval_rewards'], positions=positions, patch_artist=True,
                    labels=variant_names, widths=0.6)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title('Reward Distribution', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Reward')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=15)
    
    # 2. Food Collection Distribution
    ax = axes[0, 1]
    bp = ax.boxplot(eval_results['eval_food'], positions=positions, patch_artist=True,
                    labels=variant_names, widths=0.6)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title('Food Collection Distribution', fontsize=12, fontweight='bold')
    ax.set_ylabel('Food Collected')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=15)
    
    # 3. Success Rate Comparison
    ax = axes[1, 0]
    success_rates = eval_results['eval_success_rates']
    bars = ax.bar(positions, success_rates, color=colors, alpha=0.7, width=0.6)
    ax.set_title('Success Rate Comparison', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate')
    ax.set_xticks(positions)
    ax.set_xticklabels(variant_names, rotation=15)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Performance Summary Table
    ax = axes[1, 1]
    ax.axis('off')
    
    table_data = []
    table_data.append(['Variant', 'Avg Reward', 'Avg Food', 'Success Rate'])
    
    for idx, name in enumerate(variant_names):
        avg_reward = np.mean(eval_results['eval_rewards'][idx])
        avg_food = np.mean(eval_results['eval_food'][idx])
        success_rate = eval_results['eval_success_rates'][idx]
        
        table_data.append([
            name,
            f"{avg_reward:.1f}",
            f"{avg_food:.1f}",
            f"{success_rate:.1%}"
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.35, 0.2, 0.2, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code rows
    for i in range(1, len(table_data)):
        for j in range(4):
            table[(i, j)].set_facecolor(colors[i-1])
            table[(i, j)].set_alpha(0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved to {output_dir}/evaluation_comparison.png")


def generate_statistical_analysis(training_metrics, eval_results, output_dir):
    """Generate statistical analysis report."""
    
    report = []
    report.append("="*80)
    report.append("PRIMAL HUNT DQN ABLATION STUDY - STATISTICAL ANALYSIS")
    report.append("="*80)
    report.append("")
    
    variant_names = eval_results['variants']
    
    # Training Performance
    report.append("1. TRAINING PERFORMANCE")
    report.append("-"*80)
    report.append(f"{'Variant':<25} {'Final Reward':<15} {'Training Time':<15} {'Convergence'}")
    report.append("-"*80)
    
    for idx, name in enumerate(variant_names):
        final_rewards = training_metrics['episode_rewards'][idx][-100:]
        avg_final_reward = np.mean(final_rewards)
        training_time = training_metrics['training_times'][idx]
        
        # Check convergence (reward improvement in last 100 episodes)
        if len(training_metrics['episode_rewards'][idx]) >= 200:
            early_rewards = training_metrics['episode_rewards'][idx][-200:-100]
            late_rewards = training_metrics['episode_rewards'][idx][-100:]
            improvement = np.mean(late_rewards) - np.mean(early_rewards)
            convergence = "Yes" if improvement > 0 else "Slow"
        else:
            convergence = "N/A"
        
        report.append(f"{name:<25} {avg_final_reward:<15.2f} {training_time:<15.2f} {convergence}")
    
    report.append("")
    
    # Evaluation Performance
    report.append("2. EVALUATION PERFORMANCE (100 episodes)")
    report.append("-"*80)
    report.append(f"{'Variant':<25} {'Mean Reward':<20} {'Mean Food':<20} {'Success Rate'}")
    report.append("-"*80)
    
    for idx, name in enumerate(variant_names):
        mean_reward = np.mean(eval_results['eval_rewards'][idx])
        std_reward = np.std(eval_results['eval_rewards'][idx])
        mean_food = np.mean(eval_results['eval_food'][idx])
        std_food = np.std(eval_results['eval_food'][idx])
        success_rate = eval_results['eval_success_rates'][idx]
        
        report.append(f"{name:<25} {mean_reward:.2f} ± {std_reward:.2f}      "
                     f"{mean_food:.2f} ± {std_food:.2f}      {success_rate:.1%}")
    
    report.append("")
    
    # Key Findings
    report.append("3. KEY FINDINGS")
    report.append("-"*80)
    
    # Best performer
    best_idx = np.argmax([np.mean(r) for r in eval_results['eval_rewards']])
    report.append(f"• Best Overall Performance: {variant_names[best_idx]}")
    report.append(f"  - Average Reward: {np.mean(eval_results['eval_rewards'][best_idx]):.2f}")
    report.append(f"  - Success Rate: {eval_results['eval_success_rates'][best_idx]:.1%}")
    report.append("")
    
    # Experience replay impact
    full_dqn_idx = 0
    no_replay_idx = 1
    replay_impact = (np.mean(eval_results['eval_rewards'][full_dqn_idx]) - 
                    np.mean(eval_results['eval_rewards'][no_replay_idx]))
    report.append(f"• Experience Replay Impact: {replay_impact:+.2f} reward difference")
    report.append(f"  - Improves stability and sample efficiency")
    report.append("")
    
    # Target network impact
    no_target_idx = 2
    target_impact = (np.mean(eval_results['eval_rewards'][full_dqn_idx]) - 
                    np.mean(eval_results['eval_rewards'][no_target_idx]))
    report.append(f"• Target Network Impact: {target_impact:+.2f} reward difference")
    report.append(f"  - Reduces overestimation bias and improves stability")
    report.append("")
    
    # Vanilla Q-learning
    vanilla_idx = 3
    report.append(f"• Vanilla Q-Learning Performance:")
    report.append(f"  - Average Reward: {np.mean(eval_results['eval_rewards'][vanilla_idx]):.2f}")
    report.append(f"  - Shows importance of both replay and target network")
    report.append("")
    
    report.append("="*80)
    
    # Save report
    report_text = "\n".join(report)
    with open(os.path.join(output_dir, 'statistical_analysis.txt'), 'w') as f:
        f.write(report_text)
    
    print("\n" + report_text)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <results_directory>")
        print("Example: python evaluate.py results_20241014_120000")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    if not os.path.exists(results_dir):
        print(f"Error: Directory '{results_dir}' not found")
        sys.exit(1)
    
    evaluate_agents(results_dir, num_eval_episodes=100)
