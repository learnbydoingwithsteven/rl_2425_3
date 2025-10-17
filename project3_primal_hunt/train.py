"""
Training script for Primal Hunt DQN with ablation study.
Trains 4 variants: Full DQN, No Replay, No Target Network, Vanilla Q-Learning
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json
import os
from datetime import datetime
from tqdm import tqdm
import time

from environment import PrimalHuntEnv
from dqn_agent import DQNAgent

class TrainingMonitor:
    """Monitor and visualize training progress in real-time."""
    
    def __init__(self, num_variants=4):
        self.num_variants = num_variants
        self.variant_names = [
            'Full DQN',
            'No Replay',
            'No Target Network',
            'Vanilla Q-Learning'
        ]
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        # Metrics storage
        self.episode_rewards = [[] for _ in range(num_variants)]
        self.episode_lengths = [[] for _ in range(num_variants)]
        self.episode_food = [[] for _ in range(num_variants)]
        self.training_losses = [[] for _ in range(num_variants)]
        self.success_rates = [[] for _ in range(num_variants)]
        
        # Setup figure
        self.setup_figure()
    
    def setup_figure(self):
        """Setup matplotlib figure for real-time plotting."""
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 10))
        self.fig.suptitle('Primal Hunt DQN Training - Ablation Study', 
                         fontsize=16, fontweight='bold')
        
        # Configure subplots
        self.axes[0, 0].set_title('Episode Rewards')
        self.axes[0, 0].set_xlabel('Episode')
        self.axes[0, 0].set_ylabel('Total Reward')
        self.axes[0, 0].grid(True, alpha=0.3)
        
        self.axes[0, 1].set_title('Episode Length')
        self.axes[0, 1].set_xlabel('Episode')
        self.axes[0, 1].set_ylabel('Steps')
        self.axes[0, 1].grid(True, alpha=0.3)
        
        self.axes[0, 2].set_title('Food Collected')
        self.axes[0, 2].set_xlabel('Episode')
        self.axes[0, 2].set_ylabel('Food Amount')
        self.axes[0, 2].grid(True, alpha=0.3)
        
        self.axes[1, 0].set_title('Training Loss')
        self.axes[1, 0].set_xlabel('Update Step')
        self.axes[1, 0].set_ylabel('Loss')
        self.axes[1, 0].set_yscale('log')
        self.axes[1, 0].grid(True, alpha=0.3)
        
        self.axes[1, 1].set_title('Success Rate (100-episode window)')
        self.axes[1, 1].set_xlabel('Episode')
        self.axes[1, 1].set_ylabel('Success Rate')
        self.axes[1, 1].grid(True, alpha=0.3)
        
        self.axes[1, 2].set_title('Current Performance')
        self.axes[1, 2].axis('off')
        
        self.lines = {
            'rewards': [self.axes[0, 0].plot([], [], label=name, color=color, linewidth=2)[0] 
                       for name, color in zip(self.variant_names, self.colors)],
            'lengths': [self.axes[0, 1].plot([], [], label=name, color=color, linewidth=2)[0] 
                       for name, color in zip(self.variant_names, self.colors)],
            'food': [self.axes[0, 2].plot([], [], label=name, color=color, linewidth=2)[0] 
                    for name, color in zip(self.variant_names, self.colors)],
            'loss': [self.axes[1, 0].plot([], [], label=name, color=color, linewidth=1, alpha=0.7)[0] 
                    for name, color in zip(self.variant_names, self.colors)],
            'success': [self.axes[1, 1].plot([], [], label=name, color=color, linewidth=2)[0] 
                       for name, color in zip(self.variant_names, self.colors)]
        }
        
        for ax in [self.axes[0, 0], self.axes[0, 1], self.axes[0, 2], self.axes[1, 0], self.axes[1, 1]]:
            ax.legend(loc='best', fontsize=8)
        
        plt.tight_layout()
    
    def update(self, variant_idx, episode, reward, length, food, loss, success_rate):
        """Update metrics for a variant."""
        self.episode_rewards[variant_idx].append(reward)
        self.episode_lengths[variant_idx].append(length)
        self.episode_food[variant_idx].append(food)
        if loss is not None:
            self.training_losses[variant_idx].append(loss)
        self.success_rates[variant_idx].append(success_rate)
    
    def render(self):
        """Render current training progress."""
        for idx in range(self.num_variants):
            if len(self.episode_rewards[idx]) > 0:
                episodes = range(len(self.episode_rewards[idx]))
                
                # Update rewards
                self.lines['rewards'][idx].set_data(episodes, self.episode_rewards[idx])
                
                # Update lengths
                self.lines['lengths'][idx].set_data(episodes, self.episode_lengths[idx])
                
                # Update food
                self.lines['food'][idx].set_data(episodes, self.episode_food[idx])
                
                # Update loss
                if len(self.training_losses[idx]) > 0:
                    loss_steps = range(len(self.training_losses[idx]))
                    self.lines['loss'][idx].set_data(loss_steps, self.training_losses[idx])
                
                # Update success rate
                self.lines['success'][idx].set_data(episodes, self.success_rates[idx])
        
        # Rescale axes
        for ax in [self.axes[0, 0], self.axes[0, 1], self.axes[0, 2], self.axes[1, 0], self.axes[1, 1]]:
            ax.relim()
            ax.autoscale_view()
        
        # Update performance text
        self.axes[1, 2].clear()
        self.axes[1, 2].axis('off')
        
        text_y = 0.9
        for idx, name in enumerate(self.variant_names):
            if len(self.episode_rewards[idx]) > 0:
                recent_reward = np.mean(self.episode_rewards[idx][-10:])
                recent_food = np.mean(self.episode_food[idx][-10:])
                recent_success = self.success_rates[idx][-1] if self.success_rates[idx] else 0
                
                text = f"{name}:\n"
                text += f"  Reward: {recent_reward:.1f}\n"
                text += f"  Food: {recent_food:.1f}\n"
                text += f"  Success: {recent_success:.1%}"
                
                self.axes[1, 2].text(0.1, text_y, text, 
                                    fontsize=10, 
                                    color=self.colors[idx],
                                    verticalalignment='top',
                                    fontfamily='monospace')
                text_y -= 0.25
        
        plt.pause(0.01)
    
    def save_figure(self, filepath):
        """Save current figure."""
        self.fig.savefig(filepath, dpi=150, bbox_inches='tight')


def train_variant(env, agent, num_episodes, monitor, variant_idx, update_freq=10):
    """Train a single DQN variant."""
    success_window = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Select and execute action
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update agent
            loss = agent.update()
            
            episode_reward += reward
            episode_length += 1
            state = next_state
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Track success
        success = info.get('reason') == 'success'
        success_window.append(1 if success else 0)
        if len(success_window) > 100:
            success_window.pop(0)
        success_rate = np.mean(success_window)
        
        # Update monitor
        food_collected = env.food_collected
        monitor.update(variant_idx, episode, episode_reward, episode_length, 
                      food_collected, loss, success_rate)
        
        # Render progress
        if episode % update_freq == 0:
            monitor.render()
    
    return agent


def main():
    """Main training function."""
    # Configuration
    num_episodes = 500
    grid_size = 15
    max_steps = 200
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize monitor
    monitor = TrainingMonitor(num_variants=4)
    
    # Define variants
    variants = [
        {'name': 'Full DQN', 'use_replay': True, 'use_target_network': True},
        {'name': 'No Replay', 'use_replay': False, 'use_target_network': True},
        {'name': 'No Target Network', 'use_replay': True, 'use_target_network': False},
        {'name': 'Vanilla Q-Learning', 'use_replay': False, 'use_target_network': False}
    ]
    
    # Train each variant
    trained_agents = []
    
    for idx, variant_config in enumerate(variants):
        print(f"\n{'='*60}")
        print(f"Training Variant {idx+1}/4: {variant_config['name']}")
        print(f"{'='*60}\n")
        
        # Create environment and agent
        env = PrimalHuntEnv(grid_size=grid_size, max_steps=max_steps)
        agent = DQNAgent(
            state_dim=env.state_dim,
            action_dim=env.action_space,
            use_replay=variant_config['use_replay'],
            use_target_network=variant_config['use_target_network'],
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Train
        start_time = time.time()
        agent = train_variant(env, agent, num_episodes, monitor, idx)
        training_time = time.time() - start_time
        
        # Save agent
        agent_path = os.path.join(results_dir, f"agent_{variant_config['name'].replace(' ', '_')}.pt")
        agent.save(agent_path)
        
        trained_agents.append({
            'agent': agent,
            'config': variant_config,
            'training_time': training_time
        })
        
        print(f"\nCompleted {variant_config['name']} in {training_time:.2f} seconds")
    
    # Save final figure
    monitor.save_figure(os.path.join(results_dir, 'training_progress.png'))
    
    # Save metrics
    metrics = {
        'variants': [a['config']['name'] for a in trained_agents],
        'episode_rewards': monitor.episode_rewards,
        'episode_lengths': monitor.episode_lengths,
        'episode_food': monitor.episode_food,
        'success_rates': monitor.success_rates,
        'training_times': [a['training_time'] for a in trained_agents]
    }
    
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*60}\n")
    
    # Print summary
    print("\nFinal Performance Summary:")
    print(f"{'Variant':<25} {'Avg Reward':<15} {'Success Rate':<15} {'Training Time'}")
    print("-" * 70)
    for idx, agent_info in enumerate(trained_agents):
        avg_reward = np.mean(monitor.episode_rewards[idx][-100:])
        success_rate = monitor.success_rates[idx][-1]
        training_time = agent_info['training_time']
        print(f"{agent_info['config']['name']:<25} {avg_reward:<15.2f} {success_rate:<15.2%} {training_time:.2f}s")
    
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
