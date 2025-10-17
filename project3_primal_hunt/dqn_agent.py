"""
Deep Q-Network (DQN) Agent with Experience Replay and Target Network
Implements DQN with configurable ablation study options.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import List, Tuple

class ReplayBuffer:
    """Experience Replay Buffer for storing and sampling transitions."""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample random batch from buffer."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    """Deep Q-Network architecture."""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 128]):
        super(DQN, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class DQNAgent:
    """
    DQN Agent with configurable experience replay and target network.
    
    Ablation Study Configurations:
    - Full DQN: use_replay=True, use_target_network=True
    - No Replay: use_replay=False, use_target_network=True
    - No Target: use_replay=True, use_target_network=False
    - Vanilla Q: use_replay=False, use_target_network=False
    """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=64,
        target_update_freq=10,
        use_replay=True,
        use_target_network=True,
        device='cpu'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_replay = use_replay
        self.use_target_network = use_target_network
        self.device = device
        
        # Q-Network
        self.q_network = DQN(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Target Network
        if use_target_network:
            self.target_network = DQN(state_dim, action_dim).to(device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network.eval()
        else:
            self.target_network = self.q_network
        
        # Replay Buffer
        if use_replay:
            self.replay_buffer = ReplayBuffer(buffer_capacity)
        else:
            self.last_transition = None
        
        self.update_counter = 0
        self.training_losses = []
    
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer or as last transition."""
        if self.use_replay:
            self.replay_buffer.push(state, action, reward, next_state, done)
        else:
            self.last_transition = (state, action, reward, next_state, done)
    
    def update(self):
        """Update Q-network."""
        if self.use_replay:
            if len(self.replay_buffer) < self.batch_size:
                return None
            
            # Sample batch from replay buffer
            states, actions, rewards, next_states, dones = \
                self.replay_buffer.sample(self.batch_size)
        else:
            if self.last_transition is None:
                return None
            
            # Use only last transition
            state, action, reward, next_state, done = self.last_transition
            states = np.array([state])
            actions = np.array([action])
            rewards = np.array([reward])
            next_states = np.array([next_state])
            dones = np.array([done])
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.update_counter += 1
        if self.use_target_network and self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.training_losses.append(loss.item())
        return loss.item()
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """Save agent state."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict() if self.use_target_network else None,
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_counter': self.update_counter
        }, filepath)
    
    def load(self, filepath):
        """Load agent state."""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network'])
        if self.use_target_network and checkpoint['target_network'] is not None:
            self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.update_counter = checkpoint['update_counter']
    
    def get_config(self):
        """Get agent configuration."""
        return {
            'use_replay': self.use_replay,
            'use_target_network': self.use_target_network,
            'buffer_capacity': self.replay_buffer.buffer.maxlen if self.use_replay else 0,
            'batch_size': self.batch_size,
            'target_update_freq': self.target_update_freq,
            'gamma': self.gamma,
            'epsilon': self.epsilon
        }
