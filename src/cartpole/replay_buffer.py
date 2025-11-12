import random
from collections import deque, namedtuple
Experience = namedtuple('Experience',
                        ('state', 'action', 'reward', 'next_state', 'status'))

class ReplayBuffer:
    """
    This is the "Memory" component. It stores the agent's experiences
    and lets us sample a random batch for training.
    """

    def __init__(self, capacity):
        """Initializes the Replay Buffer.
         Args:
           -  capacity (int): The maximum number of experiences to store.
                            Once full, old memories are overwritten.
        """
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, reward, next_state, status):
        """Saves a single experience (a (s, a, r, s', d) tuple)  to the memory. """
        newExperience = Experience(state, action, reward, next_state, status)
        self.memory.append(newExperience)

    def sample(self, batch_size):
        """ Selects a random batch of experiences from memory.
        Args:
           - batch_size (int): The number of experiences to select.
        Returns:
            A list of 'Experience' namedtuples. """
        x = random.sample(self.memory, batch_size)
        return x
    
    def __len__(self):
        """Allows us to call len(buffer) to see how many experiences are currently stored."""
        return len(self.memory)