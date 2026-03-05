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

env_config = envs.env_config

def train():
    writer = SummaryWriter("runs/DQN_Experiment_1") 
    
    env = BatteryEnv(config=env_config)

    agent = DQNAgent(gamma=0.99, epsilon=1.0, lr=0.003, 
                     input_dims=103, batch_size=64, n_actions=5808, 
                     lambda_ci=0.8)

    n_episodes = 100
    scores, eps_history = [], []
    

    for i in range(n_episodes):
        score = 0

        done = False
        observation, info = env.reset()


        while not done:
            action = agent.choose_action(observation)
            dispatch_idx, plan_idx, plan_slot = decode(action)
            env_action = np.array([dispatch_idx, plan_idx, plan_slot], dtype=np.int64)
            observation_, reward, terminated, truncated, info = env.step(env_action)
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


