import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()

        # Neural Network layers
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)

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
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)

        return q_values 

    def save(self, file_name='networks.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        T.save(self.state_dict(), file_name)

class DuelingDQN(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DuelingDQN, self).__init__()
        
        # 1. Shared Feature Layer
        self.feature_layer = nn.Linear(input_dims, fc1_dims)
        
        # 2. The Value Stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )
        
        # 3. The Advantage Stream A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions)
        )

        # 4. Boilerplate (Optimizer, Loss, Device)
        self.optimiser = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.SmoothL1Loss()
        
        # Mac M4 Pro friendly device setup!
        if T.backends.mps.is_available():
            self.device = T.device('mps')
        elif T.cuda.is_available():
            self.device = T.device('cuda:0')
        else:
            self.device = T.device('cpu')
            
        self.to(self.device)

    def forward(self, state):
        features = F.relu(self.feature_layer(state))
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals

    def save(self, file_name='networks.pth'):
        model_folder_path = './models' # changed to plural to match your vault!
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        T.save(self.state_dict(), file_name)