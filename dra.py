import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, buffer_capacity=10000, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=2000, lr=0.001, t_cost_threshold=-4.5):
        self.device = torch.device("cpu")
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=buffer_capacity)
        self.T_COST_THRESHOLD = t_cost_threshold

    def select_action(self, state, env_info):
        is_in_constraint = env_info['is_in_constraint']
        T_value = env_info['T_value']
        if is_in_constraint and T_value >= self.T_COST_THRESHOLD:
            return 4  # Assertion
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1).item()

    def push(self, state, action, next_state, reward, done):
        self.memory.append(Transition(
            torch.tensor(state, dtype=torch.float32, device=self.device),
            torch.tensor([action], dtype=torch.long, device=self.device),
            torch.tensor(next_state, dtype=torch.float32, device=self.device),
            torch.tensor([reward], dtype=torch.float32, device=self.device),
            torch.tensor([done], dtype=torch.bool, device=self.device)
        ))

    def optimize_model(self, batch_size):
        if len(self.memory) < batch_size:
            return None
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat([s.unsqueeze(0) for s in batch.state])
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_mask = torch.tensor([not d.item() for d in batch.done], dtype=torch.bool, device=self.device)
        next_state_values = torch.zeros(batch_size, device=self.device)
        if non_final_mask.any():
            non_final_next_states = torch.cat([s.unsqueeze(0) for s, d in zip(batch.next_state, batch.done) if not d.item()])
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        expected_state_action_values = reward_batch + (self.gamma * next_state_values)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

class GridWorld:
    def __init__(self, rows=5, cols=5, constraint_regions=None):
        self.rows = rows
        self.cols = cols
        self.constraint_regions = set(constraint_regions) if constraint_regions else set()
        self.goal = (4, 4)
        self.T_value = 0.0
        self.T_DELTA_SUCCESS = 0.1
        self.T_DELTA_FAIL = -0.5
        self.R_DISSONANCE = -0.5
        self.R_STEP = -0.1
        self.R_GOAL = 10.0
        self.R_NOVELTY = 1.0
        self.R_FAIL = -0.5
        self.state = None

    def reset(self):
        self.state = (0, 0)
        self.T_value = 0.0
        return self.state

    def step(self, action):
        r, c = self.state
        reward = self.R_STEP
        done = False
        next_r, next_c = r, c
        is_in_C = (r, c) in self.constraint_regions

        if action < 4:
            dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
            next_r = r + dr
            next_c = c + dc
            if not (0 <= next_r < self.rows and 0 <= next_c < self.cols):
                next_r, next_c = r, c

        elif action == 4:
            if is_in_C:
                if self.T_value >= -4.5:
                    reward += self.R_NOVELTY
                    self.T_value += self.T_DELTA_SUCCESS
                else:
                    reward += self.R_FAIL
                    self.T_value += self.T_DELTA_FAIL
            else:
                reward += -0.1

        self.state = (next_r, next_c)
        
        if self.state in self.constraint_regions:
            reward += self.R_DISSONANCE
            
        if self.state == self.goal:
            reward += self.R_GOAL
            done = True
            
        self.T_value = max(-5.0, min(5.0, self.T_value))
        info = {'T_value': self.T_value, 'is_in_constraint': self.state in self.constraint_regions}
        return self.state, reward, done, info

    def get_info(self):
        return {'T_value': self.T_value, 'is_in_constraint': self.state in self.constraint_regions}

# Setup
env = GridWorld(constraint_regions=[(2,2), (3,3)])
agent = DQNAgent(state_size=2, action_size=5)
TARGET_UPDATE = 10
NUM_EPISODES = 200
BATCH_SIZE = 64
MAX_STEPS = 100

results = []

for episode in range(NUM_EPISODES):
    state = env.reset()
    info = env.get_info()
    total_reward = 0.0
    steps = 0
    done = False
    while not done and steps < MAX_STEPS:
        action = agent.select_action(state, info)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        agent.push(state, action, next_state, reward, done)
        agent.optimize_model(BATCH_SIZE)
        state = next_state
        steps += 1
    if episode % TARGET_UPDATE == 0:
        agent.update_target_net()
    if episode % 20 == 0:
        results.append(f"Episode {episode}: Total Reward = {total_reward:.1f}, Final T_value = {info['T_value']:.1f}, Steps = {steps}")

for res in results:
    print(res)
print("Training complete.")
