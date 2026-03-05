import sys
import os

#parent directory to fetch files in agents folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym 
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import envs.env_config

from agents.dqn_agent import DQNAgent
from envs.battery_env import BatteryEnv
from utils.action_encoding import encode, decode, N_ACTIONS

def train():
    writer = SummaryWriter("runs/DQN_Experiment_1") 
    
    env = gym.make('CartPole-v1')

    agent = DQNAgent(gamma=0.99, epsilon=1.0, lr=0.001, 
                 input_dims=4, batch_size=64, n_actions=2, 
                 lambda_ci=0.8, eps_dec=1e-4) # Dummy values for the last two

    n_episodes = 500
    scores, eps_history = [], []
    

    for i in range(n_episodes):
        score = 0

        done = False
        observation, info = env.reset()


        while not done:
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            score += reward

            agent.store_transition(observation, action, reward, observation_, done)

            agent.learn()
            observation = observation_
        
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])

        #Tensorboard Logging
        writer.add_scalar('Training/Episode_Reward', score, i)
        writer.add_scalar('Training/Epsilon', agent.epsilon, i)

        print('episode', i, 'score %.2f' % score, 
              'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)

    # Saving the Model
    agent.Q_eval.save('test_agent_simple_env.pth')
    writer.close()
   
if __name__ == "__main__":
    train()



