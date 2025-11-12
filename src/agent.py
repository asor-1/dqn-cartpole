import numpy as np
import time

# Setting up the environment

class GridWorld:
    def __init__(self):
        # A 4x4 grid
        # S = Start, G = Goal, H = Hole, . = Empty
        self.grid = [
            ['S', '.', '.', '.'],
            ['.', 'H', '.', '.'],
            ['.', '.', '.', 'H'],
            ['H', '.', '.', 'G']
        ]
        self.rows = 4
        self.cols = 4
        self.start_pos = (0, 0)
        self.state = self.start_pos

    def reset(self):
        """Resets the agent to the start position."""
        self.state = self.start_pos
        return self.state

    def step(self, action):
        """
        Takes an action and returns (new_state, reward, done).
        Actions: 0=Up, 1=Down, 2=Left, 3=Right
        """
        row, col = self.state

       #USING max and min to prevent the agent from going out of bounds
        #up
        if action == 0:
            row = max(0, row - 1)
        #down
        elif action == 1:
            row = min(self.rows - 1, row + 1)
        #left
        elif action == 2:
            col = max(0, col - 1)
        #right
        elif action == 3:
            col = min(self.cols - 1, col + 1)
        
        #our new sate is the new row and col
        new_state = (row, col)
        
        #Get the tile at the new_state ('G', 'H', or '.')
        tile = self.grid[row][col]

        #the reward and if the episode is 'done'
        if tile == 'G':
            reward = 1.0
            done = True
        elif tile == 'H':
            reward = -1.0
            done = True
        #having a small penalty for each step for speed
        elif tile == 'S' or tile == '.':
            reward = -0.01
            done = False

        self.state = new_state
        return new_state, reward, done

# The actual agent class

class QLearningAgent:
    def __init__(self, env):
        self.env = env
        # Q-table: 4 rows, 4 cols, 4 actions
        #remember this is a 3d grid
        self.q_table = np.zeros((env.rows, env.cols, 4))

        # Hyperparameters
        #learning rate
        self.alpha = 0.1
        #discount factor
        self.gamma = 0.99
        #The exploration rate
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def choose_action(self, state):
        """
        Chooses an action using an Epsilon-Greedy policy.
        """
        row, col = state
        
        x = np.random.rand()

        if x < self.epsilon:
            return np.random.choice([0, 1, 2, 3])
        else: 
            return np.argmax(self.q_table[row,col])

    def update_q_table(self, state, action, reward, new_state):
        """
        Updates the Q-table using the Bellman equation.
        """
        row, col = state
        new_row, new_col = new_state
        
        # This is the Bellman Equation:
        # Q(s, a) ← Q(s, a) + α * [r + γ * max(Q(s', a')) - Q(s, a)]
        # need to use row, col, action to look up the value in self.q_table
        old_q_val = self.q_table[row,col,action]
        max_future_Q = np.max(self.q_table[new_row, new_col])
        target_val = reward + self.gamma * max_future_Q
        new_q_val = old_q_val + self.alpha *(target_val - old_q_val)
        self.q_table[row,col,action] = new_q_val

    def train(self, num_episodes):
        """
        Runs the full training loop.
        """
        print("Starting training...")
        for episode in range(num_episodes):
            state = self.env.reset()
            #status of the training
            status = False
            
            while not status:
                #first need to choose an action
                act = self.choose_action(state)
                #now inputting the action into our environment and then retrieving the following values
                new_state, reward, status = self.env.step(act)
                #updating the q table
                self.update_q_table(state, act, reward, new_state)
                #updating the state
                state = new_state
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{num_episodes} - Epsilon: {self.epsilon:.3f}")
        
        print("complete training")

    def run_greedy(self):
        """Runs the agent using only the learned Q-table (no exploration)."""
        print("\n--- Running Greedy Policy ---")
        state = self.env.reset()
        done = False
        path = [state]
        
        while not done:
            row, col = state
            #if we are at #100% exploration
            action = np.argmax(self.q_table[row, col])
            #retrieving values
            state, reward, done = self.env.step(action)
            path.append(state)
            
            # Print the grid
            print(f"Action: {['Up', 'Down', 'Left', 'Right'][action]}")
            for r in range(self.env.rows):
                row_str = ""
                for c in range(self.env.cols):
                    if (r, c) == state:
                        #this is our agent
                        row_str += " A "
                    else:
                        row_str += f" {self.env.grid[r][c]} "
                print(row_str)
            print("-" * 15)
            time.sleep(0.5)
            
        if self.env.grid[state[0]][state[1]] == 'G':
            print("Goal Reached!")
        else:
            print("Ended in a hole.")
        print(f"Path: {path}")

if __name__ == "__main__":
    env = GridWorld()
    agent = QLearningAgent(env)
    
    # Run the training
    agent.train(num_episodes=1000)
    
    #result
    agent.run_greedy()
