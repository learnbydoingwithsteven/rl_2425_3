"""
Quick demo script to test the Primal Hunt environment and visualize a random agent.
"""

import numpy as np
import matplotlib.pyplot as plt
from environment import PrimalHuntEnv

def demo_random_agent(num_episodes=3):
    """Run demo with random agent."""
    env = PrimalHuntEnv(grid_size=15, max_steps=200)
    
    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}")
        print(f"{'='*60}")
        
        state = env.reset()
        done = False
        step = 0
        
        # Create figure for visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Primal Hunt - Episode {episode + 1}', fontsize=14, fontweight='bold')
        
        frames = []
        
        while not done and step < 50:  # Limit steps for demo
            # Render current state
            frame = env.render()
            frames.append(frame)
            
            # Random action
            action = np.random.randint(0, env.action_space)
            
            # Step
            next_state, reward, done, info = env.step(action)
            
            step += 1
            state = next_state
            
            # Print info every 10 steps
            if step % 10 == 0:
                env_info = env.get_info()
                print(f"Step {step}: Reward={reward:.1f}, Food={env_info['food_collected']}, "
                      f"Weather={env_info['weather']}")
        
        # Final info
        env_info = env.get_info()
        print(f"\nEpisode finished!")
        print(f"Total steps: {env_info['step']}")
        print(f"Food collected: {env_info['food_collected']}")
        print(f"Total reward: {env_info['total_reward']:.2f}")
        print(f"Reason: {info.get('reason', 'unknown')}")
        
        # Visualize start, middle, end
        if len(frames) >= 3:
            indices = [0, len(frames)//2, -1]
            titles = ['Start', 'Middle', 'End']
            
            for ax, idx, title in zip(axes, indices, titles):
                ax.imshow(frames[idx])
                ax.set_title(title)
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(f'demo_episode_{episode+1}.png', dpi=100, bbox_inches='tight')
            plt.close()
            print(f"Visualization saved to demo_episode_{episode+1}.png")
        
        # Legend
        print("\nEnvironment Legend:")
        print("  🔵 Blue = Hunter")
        print("  🟢 Green = Food")
        print("  🟡 Yellow = Friendly Animal")
        print("  🟠 Orange = Hostile Animal")
        print("  🔴 Red = Hostile Tribe")
        print("  ⬜ Gray = Obstacle")


def visualize_environment_layout():
    """Visualize a single environment layout."""
    env = PrimalHuntEnv(grid_size=15, max_steps=200)
    state = env.reset()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Render
    grid = env.render()
    ax.imshow(grid)
    ax.set_title('Primal Hunt Environment Layout', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label='Hunter'),
        Patch(facecolor='green', label='Food'),
        Patch(facecolor='yellow', label='Friendly Animal'),
        Patch(facecolor='orange', label='Hostile Animal'),
        Patch(facecolor='red', label='Hostile Tribe'),
        Patch(facecolor='gray', label='Obstacle')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
    
    # Add info text
    info = env.get_info()
    info_text = f"Grid Size: {env.grid_size}x{env.grid_size}\n"
    info_text += f"Max Steps: {env.max_steps}\n"
    info_text += f"Food Sources: {info['num_food_remaining']}\n"
    info_text += f"Animals: {len(env.animal_positions)}\n"
    info_text += f"Tribes: {len(env.tribe_positions)}\n"
    info_text += f"Obstacles: {len(env.obstacles)}"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('environment_layout.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Environment layout saved to environment_layout.png")
    print("\nEnvironment Details:")
    print(f"  Grid Size: {env.grid_size}x{env.grid_size}")
    print(f"  State Dimension: {env.state_dim}")
    print(f"  Action Space: {env.action_space}")
    print(f"  Max Steps: {env.max_steps}")
    print(f"  Food Sources: {info['num_food_remaining']}")
    print(f"  Animals: {len(env.animal_positions)} ({sum(env.animal_hostile)} hostile)")
    print(f"  Tribes: {len(env.tribe_positions)}")
    print(f"  Obstacles: {len(env.obstacles)}")


if __name__ == "__main__":
    print("="*60)
    print("PRIMAL HUNT ENVIRONMENT DEMO")
    print("="*60)
    
    # Visualize environment
    print("\n1. Generating environment layout...")
    visualize_environment_layout()
    
    # Run random agent demo
    print("\n2. Running random agent demo...")
    demo_random_agent(num_episodes=3)
    
    print("\n" + "="*60)
    print("Demo complete! Check the generated PNG files.")
    print("="*60)
