"""
Primal Hunt Environment
A primitive hunter gathering food in a dangerous environment with animals, tribes, and obstacles.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import random

class PrimalHuntEnv:
    """
    Grid-based environment where a primitive hunter must gather food while avoiding dangers.
    
    State Space:
    - Hunter position (x, y)
    - Food locations and amounts
    - Animal positions and states
    - Tribe positions
    - Obstacle locations
    - Weather condition
    
    Action Space:
    - 0: Move Up
    - 1: Move Down
    - 2: Move Left
    - 3: Move Right
    - 4: Stay (gather food if at food location)
    
    Rewards:
    - +10 for gathering food
    - -20 for encountering hostile animal
    - -30 for encountering hostile tribe
    - -1 for each step (survival cost)
    - +50 for completing episode with enough food
    """
    
    def __init__(self, grid_size=15, max_steps=200, dynamic=False):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.dynamic = dynamic  # For variant with moving threats
        
        # Action space
        self.action_space = 5
        self.actions = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1),   # Right
            4: (0, 0)    # Stay/Gather
        }
        
        # State space dimensions
        self.state_dim = 8  # [hunter_x, hunter_y, nearest_food_dist, nearest_animal_dist, 
                            # nearest_tribe_dist, food_collected, steps_remaining, weather]
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = 0
        self.food_collected = 0
        self.total_reward = 0
        
        # Place hunter in safe starting position
        self.hunter_pos = np.array([self.grid_size // 2, self.grid_size // 2])
        
        # Generate food locations (5-8 food sources)
        num_food = random.randint(5, 8)
        self.food_locations = []
        self.food_amounts = []
        for _ in range(num_food):
            pos = self._get_random_position(exclude=[self.hunter_pos])
            self.food_locations.append(pos)
            self.food_amounts.append(random.randint(1, 3))
        
        # Generate animals (3-5 animals)
        num_animals = random.randint(3, 5)
        self.animal_positions = []
        self.animal_hostile = []
        for _ in range(num_animals):
            pos = self._get_random_position(exclude=[self.hunter_pos] + self.food_locations)
            self.animal_positions.append(pos)
            self.animal_hostile.append(random.random() < 0.6)  # 60% hostile
        
        # Generate tribes (2-3 tribes)
        num_tribes = random.randint(2, 3)
        self.tribe_positions = []
        for _ in range(num_tribes):
            pos = self._get_random_position(
                exclude=[self.hunter_pos] + self.food_locations + self.animal_positions
            )
            self.tribe_positions.append(pos)
        
        # Generate obstacles (10-15 obstacles)
        num_obstacles = random.randint(10, 15)
        self.obstacles = []
        for _ in range(num_obstacles):
            pos = self._get_random_position(
                exclude=[self.hunter_pos] + self.food_locations + 
                        self.animal_positions + self.tribe_positions
            )
            self.obstacles.append(pos)
        
        # Weather: 0=clear, 1=rain, 2=storm
        self.weather = 0
        
        return self._get_state()
    
    def _get_random_position(self, exclude=[]) -> np.ndarray:
        """Get random position not in exclude list."""
        while True:
            pos = np.array([random.randint(0, self.grid_size-1), 
                          random.randint(0, self.grid_size-1)])
            if not any(np.array_equal(pos, ex) for ex in exclude):
                return pos
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation."""
        # Nearest food distance
        if len(self.food_locations) > 0:
            food_dists = [np.linalg.norm(self.hunter_pos - food) 
                         for food in self.food_locations]
            nearest_food = min(food_dists) / (self.grid_size * np.sqrt(2))
        else:
            nearest_food = 1.0
        
        # Nearest animal distance
        if len(self.animal_positions) > 0:
            animal_dists = [np.linalg.norm(self.hunter_pos - animal) 
                           for animal in self.animal_positions]
            nearest_animal = min(animal_dists) / (self.grid_size * np.sqrt(2))
        else:
            nearest_animal = 1.0
        
        # Nearest tribe distance
        if len(self.tribe_positions) > 0:
            tribe_dists = [np.linalg.norm(self.hunter_pos - tribe) 
                          for tribe in self.tribe_positions]
            nearest_tribe = min(tribe_dists) / (self.grid_size * np.sqrt(2))
        else:
            nearest_tribe = 1.0
        
        state = np.array([
            self.hunter_pos[0] / self.grid_size,
            self.hunter_pos[1] / self.grid_size,
            nearest_food,
            nearest_animal,
            nearest_tribe,
            self.food_collected / 10.0,  # Normalize
            (self.max_steps - self.current_step) / self.max_steps,
            self.weather / 2.0
        ], dtype=np.float32)
        
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return next state, reward, done, info."""
        self.current_step += 1
        reward = -1  # Step penalty
        done = False
        info = {'reason': ''}
        
        # Move hunter
        if action < 4:  # Movement action
            move = self.actions[action]
            new_pos = self.hunter_pos + move
            
            # Check boundaries
            if (0 <= new_pos[0] < self.grid_size and 
                0 <= new_pos[1] < self.grid_size):
                
                # Check obstacles
                if not any(np.array_equal(new_pos, obs) for obs in self.obstacles):
                    self.hunter_pos = new_pos
        
        # Check for food gathering (action 4 or at food location)
        for i, food_pos in enumerate(self.food_locations):
            if np.array_equal(self.hunter_pos, food_pos) and self.food_amounts[i] > 0:
                if action == 4:  # Gather action
                    amount = self.food_amounts[i]
                    self.food_collected += amount
                    self.food_amounts[i] = 0
                    reward += 10 * amount
                    info['food_gathered'] = amount
        
        # Check for animal encounters
        for i, animal_pos in enumerate(self.animal_positions):
            if np.linalg.norm(self.hunter_pos - animal_pos) < 1.5:
                if self.animal_hostile[i]:
                    reward -= 20
                    info['animal_encounter'] = 'hostile'
                else:
                    reward += 5  # Friendly animal
                    info['animal_encounter'] = 'friendly'
        
        # Check for tribe encounters
        for tribe_pos in self.tribe_positions:
            if np.linalg.norm(self.hunter_pos - tribe_pos) < 2.0:
                reward -= 30
                info['tribe_encounter'] = True
                done = True
                info['reason'] = 'hostile_tribe'
        
        # Dynamic environment updates
        if self.dynamic and self.current_step % 10 == 0:
            self._update_environment()
        
        # Check termination conditions
        if self.current_step >= self.max_steps:
            done = True
            if self.food_collected >= 10:
                reward += 50
                info['reason'] = 'success'
            else:
                info['reason'] = 'timeout'
        
        next_state = self._get_state()
        self.total_reward += reward
        
        return next_state, reward, done, info
    
    def _update_environment(self):
        """Update environment for dynamic variant."""
        # Move animals randomly
        for i in range(len(self.animal_positions)):
            move = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)])
            new_pos = self.animal_positions[i] + move
            if (0 <= new_pos[0] < self.grid_size and 
                0 <= new_pos[1] < self.grid_size):
                self.animal_positions[i] = new_pos
        
        # Change weather
        if random.random() < 0.1:
            self.weather = random.randint(0, 2)
    
    def render(self, mode='human'):
        """Render the environment."""
        grid = np.zeros((self.grid_size, self.grid_size, 3))
        
        # Draw obstacles (gray)
        for obs in self.obstacles:
            grid[obs[0], obs[1]] = [0.5, 0.5, 0.5]
        
        # Draw food (green)
        for i, food in enumerate(self.food_locations):
            if self.food_amounts[i] > 0:
                intensity = self.food_amounts[i] / 3.0
                grid[food[0], food[1]] = [0, intensity, 0]
        
        # Draw animals (yellow=friendly, orange=hostile)
        for i, animal in enumerate(self.animal_positions):
            if self.animal_hostile[i]:
                grid[animal[0], animal[1]] = [1, 0.5, 0]  # Orange
            else:
                grid[animal[0], animal[1]] = [1, 1, 0]  # Yellow
        
        # Draw tribes (red)
        for tribe in self.tribe_positions:
            grid[tribe[0], tribe[1]] = [1, 0, 0]
        
        # Draw hunter (blue)
        grid[self.hunter_pos[0], self.hunter_pos[1]] = [0, 0, 1]
        
        return grid
    
    def get_info(self) -> Dict:
        """Get current environment information."""
        return {
            'step': self.current_step,
            'food_collected': self.food_collected,
            'total_reward': self.total_reward,
            'hunter_pos': self.hunter_pos.tolist(),
            'num_food_remaining': sum(1 for amt in self.food_amounts if amt > 0),
            'weather': ['clear', 'rain', 'storm'][self.weather]
        }
