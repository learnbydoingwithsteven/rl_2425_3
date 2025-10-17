# Project 3: Primal Hunt - DQN Ablation Study

## Overview

This project implements a Deep Q-Network (DQN) agent for the **Primal Hunt** environment, where a primitive hunter must gather food while avoiding dangerous animals, hostile tribes, and obstacles. The project includes a comprehensive ablation study to evaluate the importance of **Experience Replay** and **Target Networks** in DQN.

## Environment: Primal Hunt

### Description
A grid-based environment (15×15) where a primitive hunter navigates to collect food while avoiding threats.

### State Space (8 dimensions)
- Hunter position (x, y) - normalized
- Distance to nearest food
- Distance to nearest animal
- Distance to nearest tribe
- Food collected
- Steps remaining
- Weather condition

### Action Space (5 actions)
- 0: Move Up
- 1: Move Down
- 2: Move Left
- 3: Move Right
- 4: Stay/Gather (collect food if at food location)

### Rewards
- **+10 per unit** for gathering food
- **-20** for hostile animal encounter
- **-30** for hostile tribe encounter (episode ends)
- **-1** per step (survival cost)
- **+50** bonus for completing episode with ≥10 food

### Environment Elements
- **Food Sources**: 5-8 locations with 1-3 units each (green)
- **Animals**: 3-5 animals, 60% hostile (yellow=friendly, orange=hostile)
- **Tribes**: 2-3 hostile tribes (red)
- **Obstacles**: 10-15 impassable obstacles (gray)
- **Hunter**: Blue marker

## DQN Implementation

### Architecture
- **Neural Network**: 2 hidden layers (128 units each) with ReLU activation
- **Input**: 8-dimensional state vector
- **Output**: Q-values for 5 actions

### Hyperparameters
- Learning Rate: 0.001
- Discount Factor (γ): 0.99
- Epsilon: 1.0 → 0.01 (decay: 0.995)
- Replay Buffer: 10,000 transitions
- Batch Size: 64
- Target Network Update: Every 10 episodes

## Ablation Study

Four variants are trained and evaluated:

| Variant | Experience Replay | Target Network |
|---------|------------------|----------------|
| **Full DQN** | ✓ | ✓ |
| **No Replay** | ✗ | ✓ |
| **No Target Network** | ✓ | ✗ |
| **Vanilla Q-Learning** | ✗ | ✗ |

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Train All Variants

```bash
python train.py
```

This will:
- Train all 4 DQN variants for 500 episodes each
- Display real-time training progress with 6 subplots:
  - Episode Rewards
  - Episode Length
  - Food Collected
  - Training Loss
  - Success Rate (100-episode window)
  - Current Performance Summary
- Save trained agents and metrics to `results_TIMESTAMP/`

### 2. Evaluate Trained Agents

```bash
python evaluate.py results_TIMESTAMP
```

This will:
- Evaluate each agent for 100 episodes
- Generate animated GIFs of sample episodes
- Create comprehensive comparison plots
- Produce statistical analysis report

## Output Files

### Training Output (`results_TIMESTAMP/`)
- `agent_Full_DQN.pt` - Trained Full DQN agent
- `agent_No_Replay.pt` - Trained No Replay agent
- `agent_No_Target_Network.pt` - Trained No Target Network agent
- `agent_Vanilla_Q-Learning.pt` - Trained Vanilla Q-Learning agent
- `metrics.json` - Training metrics for all variants
- `training_progress.png` - Real-time training visualization

### Evaluation Output (`results_TIMESTAMP/evaluation/`)
- `Full_DQN_episode.gif` - Animated episode visualization
- `No_Replay_episode.gif` - Animated episode visualization
- `No_Target_Network_episode.gif` - Animated episode visualization
- `Vanilla_Q-Learning_episode.gif` - Animated episode visualization
- `evaluation_results.json` - Evaluation metrics
- `evaluation_comparison.png` - Comprehensive comparison plots
- `statistical_analysis.txt` - Statistical analysis report

## Visualization Features

### Real-Time Training Dashboard
- **6 synchronized plots** updating during training
- **Color-coded variants** for easy comparison
- **Performance metrics** displayed in real-time
- **Smooth animations** with automatic axis scaling

### Evaluation Visualizations
1. **Reward Distribution** - Box plots comparing reward distributions
2. **Food Collection** - Box plots comparing food collection
3. **Success Rate** - Bar chart with percentage labels
4. **Performance Table** - Summary statistics for all variants

### Episode Animations
- **Grid visualization** showing hunter, food, animals, tribes, obstacles
- **Color-coded elements** for easy identification
- **Step counter** showing progress
- **Legend** explaining all elements

## Expected Results

### Full DQN (Best Performance)
- Highest average reward
- Most stable learning
- Best sample efficiency
- Highest success rate

### No Replay
- More unstable learning
- Lower sample efficiency
- Susceptible to catastrophic forgetting
- Moderate performance

### No Target Network
- Overestimation bias
- Less stable Q-values
- Oscillating performance
- Moderate performance

### Vanilla Q-Learning (Worst Performance)
- Most unstable learning
- Lowest sample efficiency
- Poor convergence
- Lowest success rate

## Key Findings

The ablation study demonstrates:

1. **Experience Replay** significantly improves:
   - Sample efficiency
   - Learning stability
   - Final performance

2. **Target Network** significantly improves:
   - Q-value stability
   - Reduces overestimation bias
   - Smoother convergence

3. **Combined Effect**: Both components are crucial for optimal DQN performance

## Project Structure

```
project3_primal_hunt/
├── environment.py          # Primal Hunt environment implementation
├── dqn_agent.py           # DQN agent with ablation configurations
├── train.py               # Training script with real-time visualization
├── evaluate.py            # Evaluation and analysis script
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── results_TIMESTAMP/    # Training results (generated)
    ├── agent_*.pt        # Trained agents
    ├── metrics.json      # Training metrics
    ├── training_progress.png
    └── evaluation/       # Evaluation results
        ├── *_episode.gif
        ├── evaluation_comparison.png
        ├── evaluation_results.json
        └── statistical_analysis.txt
```

## Technical Details

### DQN Algorithm
1. Initialize Q-network and target network
2. Initialize replay buffer
3. For each episode:
   - Select action using ε-greedy policy
   - Execute action and observe reward, next state
   - Store transition in replay buffer
   - Sample random batch from buffer
   - Compute target: r + γ max Q_target(s', a')
   - Update Q-network to minimize (Q(s,a) - target)²
   - Periodically update target network
   - Decay ε

### Ablation Modifications
- **No Replay**: Use only last transition for updates
- **No Target**: Use Q-network for both current and target Q-values
- **Vanilla**: Combine both modifications

## Performance Metrics

- **Episode Reward**: Total reward accumulated in episode
- **Episode Length**: Number of steps taken
- **Food Collected**: Amount of food gathered
- **Success Rate**: Percentage of episodes ending with ≥10 food
- **Training Loss**: MSE between predicted and target Q-values

## Future Extensions

1. **Dynamic Environment Variant**: Moving animals and changing weather
2. **Prioritized Experience Replay**: Sample important transitions more frequently
3. **Double DQN**: Reduce overestimation bias further
4. **Dueling DQN**: Separate value and advantage streams
5. **Multi-agent**: Multiple hunters competing or cooperating

## References

- Mnih et al. (2015). "Human-level control through deep reinforcement learning." Nature.
- Van Hasselt et al. (2016). "Deep Reinforcement Learning with Double Q-learning." AAAI.
- Schaul et al. (2016). "Prioritized Experience Replay." ICLR.

## Author

Reinforcement Learning Course Project 2024-25

## License

MIT License
