import numpy as np
import random
from collections import deque



class ReplayBuffer:

    def __init__(self, max_size):
        self.memory = deque(maxlen=max_size)
        
    
    def add_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        

    def sample_batch(self, batch_size):
        batch = random.sample(self.memory, batch_size)

        # the batch needs to be seperated back into individual arrays
        states, actions, rewards, next_states, dones = zip(*batch)

        return(np.array(states),
        np.array(actions),
        np.array(rewards, dtype=np.float32),
        np.array(next_states),
        np.array(dones, dtype=np.bool_))
    
    def __len__(self):
        return len(self.memory)
        
        