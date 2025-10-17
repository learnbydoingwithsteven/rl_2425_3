"""
Quick test script to verify all components work correctly.
Runs a minimal version of training (10 episodes) to test the pipeline.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from environment import PrimalHuntEnv
from dqn_agent import DQNAgent

def test_environment():
    """Test environment functionality."""
    print("Testing Environment...")
    env = PrimalHuntEnv(grid_size=15, max_steps=200)
    
    # Test reset
    state = env.reset()
    assert state.shape == (8,), f"Expected state shape (8,), got {state.shape}"
    
    # Test step
    action = 0
    next_state, reward, done, info = env.step(action)
    assert next_state.shape == (8,), f"Expected next_state shape (8,), got {next_state.shape}"
    assert isinstance(reward, (int, float)), f"Expected reward to be numeric, got {type(reward)}"
    assert isinstance(done, bool), f"Expected done to be bool, got {type(done)}"
    assert isinstance(info, dict), f"Expected info to be dict, got {type(info)}"
    
    # Test render
    grid = env.render()
    assert grid.shape == (15, 15, 3), f"Expected grid shape (15, 15, 3), got {grid.shape}"
    
    print("✓ Environment tests passed!")
    return True

def test_dqn_agent():
    """Test DQN agent functionality."""
    print("\nTesting DQN Agent...")
    
    # Test all variants
    variants = [
        {'name': 'Full DQN', 'use_replay': True, 'use_target_network': True},
        {'name': 'No Replay', 'use_replay': False, 'use_target_network': True},
        {'name': 'No Target', 'use_replay': True, 'use_target_network': False},
        {'name': 'Vanilla', 'use_replay': False, 'use_target_network': False}
    ]
    
    for variant in variants:
        agent = DQNAgent(
            state_dim=8,
            action_dim=5,
            use_replay=variant['use_replay'],
            use_target_network=variant['use_target_network'],
            device='cpu'
        )
        
        # Test action selection
        state = np.random.randn(8)
        action = agent.select_action(state)
        assert 0 <= action < 5, f"Invalid action {action}"
        
        # Test store transition
        next_state = np.random.randn(8)
        agent.store_transition(state, action, 1.0, next_state, False)
        
        # Test update
        if variant['use_replay']:
            # Need enough samples for replay
            for _ in range(100):
                agent.store_transition(state, action, 1.0, next_state, False)
        
        loss = agent.update()
        if loss is not None:
            assert isinstance(loss, float), f"Expected loss to be float, got {type(loss)}"
        
        print(f"  ✓ {variant['name']} tests passed!")
    
    print("✓ All DQN agent tests passed!")
    return True

def test_training_loop():
    """Test minimal training loop."""
    print("\nTesting Training Loop (10 episodes)...")
    
    env = PrimalHuntEnv(grid_size=15, max_steps=50)  # Shorter episodes for testing
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_space,
        use_replay=True,
        use_target_network=True,
        device='cpu'
    )
    
    rewards = []
    
    for episode in range(10):
        state = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 50:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            
            episode_reward += reward
            state = next_state
            steps += 1
        
        agent.decay_epsilon()
        rewards.append(episode_reward)
        
        if (episode + 1) % 5 == 0:
            print(f"  Episode {episode+1}/10: Reward={episode_reward:.1f}, Steps={steps}")
    
    print(f"\n  Average Reward: {np.mean(rewards):.2f}")
    print("✓ Training loop tests passed!")
    return True

def test_visualization():
    """Test visualization functionality."""
    print("\nTesting Visualization...")
    
    env = PrimalHuntEnv(grid_size=15, max_steps=200)
    state = env.reset()
    
    # Create simple plot
    fig, ax = plt.subplots(figsize=(8, 8))
    grid = env.render()
    ax.imshow(grid)
    ax.set_title('Environment Test')
    ax.axis('off')
    
    plt.savefig('test_visualization.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Visualization saved to test_visualization.png")
    print("✓ Visualization tests passed!")
    return True

def main():
    """Run all tests."""
    print("="*60)
    print("  PROJECT 3: PRIMAL HUNT - QUICK TEST")
    print("="*60)
    print("\nRunning component tests...\n")
    
    try:
        # Test each component
        test_environment()
        test_dqn_agent()
        test_training_loop()
        test_visualization()
        
        print("\n" + "="*60)
        print("  ✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nThe project is ready to run!")
        print("\nNext steps:")
        print("  1. Run demo: python demo.py")
        print("  2. Run training: python train.py")
        print("  3. Run evaluation: python evaluate.py results_TIMESTAMP")
        print("  4. Or run complete pipeline: python run_complete_project.py")
        print("\n" + "="*60)
        
        return True
        
    except Exception as e:
        print("\n" + "="*60)
        print("  ❌ TEST FAILED!")
        print("="*60)
        print(f"\nError: {str(e)}")
        print("\nPlease check the error message and fix the issue.")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
