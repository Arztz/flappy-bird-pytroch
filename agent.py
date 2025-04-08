import argparse
import datetime
import flappy_bird_gymnasium
import gymnasium
from matplotlib import pyplot as plt
import numpy as np
from dqn import DQN
import torch
from experience_replay import ReplayMemory
import itertools
import yaml
import random
import torch.nn as nn
import os
import matplotlib
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

matplotlib.use('Agg')

device = "cuda" if torch.cuda.is_available() else "cpu"
class Agent:
    def __init__(self, hyperparameters_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameters_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameters_sets[hyperparameters_set]

        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.discount_factor_g = hyperparameters['discount_factor_g']
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.fc1_nodes = hyperparameters['fc1_nodes']
        self.env_make_params = hyperparameters.get('env_make_params',{})

        self.loss_fn = nn.MSELoss()
        self.optimizer = None


        self.LOG_FILE = os.path.join(RUNS_DIR, f'{hyperparameters_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{hyperparameters_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{hyperparameters_set}.png')

    def run(self,is_training=True,render=False):

        if is_training:
            start_time = datetime.datetime.now()
            last_graph_update_time = start_time
            log_message = f"Start time: {start_time}\n"
            print(log_message)
            with open(self.LOG_FILE, 'w') as log_file:
                log_file.write(log_message+ '\n')

        # env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)
        

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        reward_per_episode = []
        epsilon_history = []

        policy_dqn = DQN(num_states,num_actions,self.fc1_nodes ).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init
            
            target_dqn = DQN(num_states,num_actions,self.fc1_nodes  ).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            step_count = 0

            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
            epsilon_history = []
            best_reward = -9999999
        else:
            # Load the model
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        #  Training loop
        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state,dtype=torch.float,device=device)

            terminated = False
            episode_reward = 0.0

            while (not terminated and episode_reward < self.stop_on_reward):
                # Next action:
                # (feed the observation to your agent here)
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action,dtype=torch.long,device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Processing:
                new_state, reward, terminated, _, info = env.step(action.item())
                
                episode_reward += reward

                new_state = torch.tensor(new_state,dtype=torch.float,device=device)
                reward = torch.tensor(reward,dtype=torch.float,device=device)
                

                if is_training:
                    memory.append((state,action,new_state,reward,terminated))
                    step_count += 1

                state = new_state

            reward_per_episode.append(episode_reward)

            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.datetime.now().strftime(DATE_FORMAT)}: New best reward: {episode_reward} at episode {episode}"
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as log_file:
                        log_file.write(log_message + '\n')
                    best_reward = episode_reward

                current_time = datetime.datetime.now()
                if current_time - last_graph_update_time > datetime.timedelta(seconds=10):
                    self.save_graph(reward_per_episode, epsilon_history)
                    last_graph_update_time = current_time


                if len(memory) > self.mini_batch_size:

                    #sample from memory
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch,policy_dqn,target_dqn)
                    epsilon = max(epsilon * self.epsilon_decay,self.epsilon_min)
                    epsilon_history.append(epsilon)
                    if step_count > self.network_sync_rate == 0:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0
    def save_graph(self,reward_per_episode, epsilon_history):
        fig = plt.figure(1)

        mean_reward = np.zeros(len(reward_per_episode))

        for i in range(len(mean_reward)):
            mean_reward[i] = np.mean(reward_per_episode[max(0,i-99):i+1])
        plt.subplot(121)
        plt.ylabel('Mean Reward')
        plt.plot(mean_reward)

        plt.subplot(122)
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)


    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states,actions,new_states,rewards,terminations = zip(*mini_batch)

        states = torch.stack(states).to(device)
        actions = torch.stack(actions).to(device)   
        new_states = torch.stack(new_states).to(device) 
        rewards = torch.stack(rewards).to(device)   
        terminations = torch.tensor(terminations).float().to(device)
        with torch.no_grad():
            target_q = rewards + (1-terminations )* self.discount_factor_g  * target_dqn(new_states).max(dim=1)[0]


        current_q = policy_dqn(states).gather(dim=1,index=actions.unsqueeze(dim=1)).squeeze()

        loss = self.loss_fn(current_q,target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test model')
    parser.add_argument('hyperparameters',help='')
    parser.add_argument('--train',help='Training mode',action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameters_set=args.hyperparameters)
    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False,render=True)

    agent = Agent('cartpole1')
    agent.run(is_training=True,render=True)
    # agent.run(is_training=False,render=True)