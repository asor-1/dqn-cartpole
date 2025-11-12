import gymnasium as gym
import math
import random
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from qn_model import DQN
from replay_buffer import ReplayBuffer, Experience
env = gym.make("CartPole-v1")
# Get the number of actions and observations from the environment
n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

#how many memories to train at once
"""Notes:
If I increase BATCH_SIZE it will slow down training and use more compute
If I decrease BATCH_SIZE it will speed up training but more unstable"""
BATCH_SIZE = 128
#Discount factor
GAMMA = 0.99
#Starting value for high exploration
EPS_START = 0.9
#End value for low exploration
EPS_END = 0.05
#How fast the epsilon decays
"""Notes:
If I increase the decay it will take longer to get out of exploring stage"""
EPS_DECAY = 1000
#The update rate for the target network
"""Notes:
Slowly updates the weights"""
TAU = 0.005
#Learning rate for Adam
""" Notes:
Changed from 1e-4 to 1e-4 because it wasn't learning fast enough """
LR = 1e-3

#create one for the agent's policy and one for the stable "target"
policy_net = DQN(n_observations, n_actions)
target_net = DQN(n_observations, n_actions)
# Copy the weights from the policy_net to the target_net
target_net.load_state_dict(policy_net.state_dict())

# 2. Create the "Memory"
# We initialize it with a capacity of 10,000 memories
memory = ReplayBuffer(10000)

# 3. Create the Optimizer
# This will *only* train the policy_net
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
episode_durations = []
# We'll need this to calculate the decaying epsilon
steps_done = 0

def choose_action(state):
    """ Chooses an action using an Epsilon-Greedy policy."""
    global steps_done
    
    #calculate the current epsilon value
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    #means we explot
    if sample > eps_threshold:
        with torch.no_grad():
            #state is already a tensor
            q_vals = policy_net(state)
            return q_vals.argmax().view(1, 1)
    #explore
    else:
        #random action
        action = env.action_space.sample()
        return torch.tensor([[action]], dtype=torch.long)

def optimize_model():
    """ This is the core training function. It samples a batch from memory and performs one step of gradient descent.
    """
    #just base level check
    if len(memory) < BATCH_SIZE :
        return
    
    #now lets sample a batch of experiences
    transitions = memory.sample(BATCH_SIZE)
    unzip_batch = Experience(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, unzip_batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in unzip_batch.next_state if s is not None])
    
    # Concatenate (stack) all the states, actions, and rewards from the batch
    state_batch = torch.cat(unzip_batch.state)
    action_batch = torch.cat(unzip_batch.action)
    reward_batch = torch.cat(unzip_batch.reward)
    #calculating the predictions
    all_q = policy_net(state_batch).gather(1, action_batch)
    #calculate next state
    next_state = torch.zeros(BATCH_SIZE)
    #dont need gradients
    with torch.no_grad():
        next_state[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    
    #calculate target
    target = reward_batch + (GAMMA * next_state) 

    criterion = nn.SmoothL1Loss()
    #loss
    loss = criterion(all_q, target.unsqueeze(1))
    #gradient descent

    #clear the old gradients
    optimizer.zero_grad()
    #new gradients
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    #optimizer updates the policy nets weights
    optimizer.step()


if __name__ == "__main__":
    
    num_episodes = 1000
    for i_episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        done = False
        t = 0 # Step counter for this episode
        while not done:
            action = choose_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            t += 1

            reward = torch.tensor([reward])
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

            memory.push(state, action, reward, next_state, terminated)
            
            # Move to the next state
            state = next_state

            # Perform one step of optimization (training)
            optimize_model()

        # Update the Target Network (soft update)
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)
        
        episode_durations.append(t + 1)
        # This will now print the duration!
        if i_episode % 50 == 0:
            print(f"Episode {i_episode} complete. Duration: {t+1}")

    print('Complete training')
    
    # print("\n Running Final Policy (Greedy)")
    
    # # Re-initialize environment for rendering
    # env = gym.make("CartPole-v1", render_mode="human")
    # state, info = env.reset()
    # state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    # done = False
    
    # while not done:
    #     # Run in 'exploit' mode (no epsilon)
    #     with torch.no_grad():
    #         q_vals = policy_net(state)
    #         action = q_vals.argmax().view(1, 1)
        
    #     observation, reward, terminated, truncated, _ = env.step(action.item())
    #     done = terminated or truncated
        
    #     if not done:
    #         state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            
    #     time.sleep(0.02) # Slow down the rendering

    # env.close()

    # Plot
    plt.figure(1)
    plt.title('Result')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(episode_durations)
    plt.show()