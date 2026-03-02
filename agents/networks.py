import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os 

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims=103, fc1_dims=256, fc2_dims=256, n_actions=5808):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims 
        self.fc1_dims = fc1_dims 
        self.fc2_dims = fc2_dims 
        self.n_actions = n_actions 
        
        # Neural Network layers
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        # Optimizer 
        self.optimiser = optim.Adam(self.parameters(), lr=lr)

        # Loss function
        # (Huber Loss) because it prevents crazy gradients if the agent makes a huge mistake early on.
        self.loss = nn.SmoothL1Loss()

        # Device mapping
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    # Handling forward propagation
    def forward(self, state):
        x = F.relu(self.fc1(T.from_numpy(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)

        return q_values 

    def save(self, file_name='networks.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        T.save(self.state_dict(), file_name)