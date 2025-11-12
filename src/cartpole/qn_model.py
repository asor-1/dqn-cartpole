import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    This is our "Brain" (the Function Approximator).
    It's a simple neural network that takes a state and
    outputs the Q-values for each possible action.
    """

    def __init__(self, n_observations, n_actions):
        """Initializes the network layers.
        Args:
            n_observations (int): The size of the state space (4 for CartPole)
            n_actions (int): The number of possible actions (2 for CartPole) """
        super(DQN, self).__init__() # Calls the constructor of the parent nn.Module class
        #number of outputs should be 128
        self.layer1 = nn.Linear(n_observations, 128)
        #number of outputs should also be 128
        self.layer2 = nn.Linear(128, 128)
        #n_actions
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        """
        This is the "forward pass" — it defines how a state (x) flows through the network to produce Q-values. 
        Args:
            x (torch.Tensor): A batch of states.    
        Returns:
            torch.Tensor: The Q-values for each action for each state in the batch.
        """
        x= F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)