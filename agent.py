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
import psutil
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

matplotlib.use('Agg')

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(18)
class Agent:
    def __init__(self, hyperparameters_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameters_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameters_sets[hyperparameters_set]
        self.env_id             = hyperparameters['env_id']
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
        self.enable_double_dqn = hyperparameters['enable_double_dqn']
        self.enable_dueling_dqn = hyperparameters['enable_dueling_dqn']
        self.pretrained_model = hyperparameters.get('pretrained_model', None)

        self.loss_fn = nn.MSELoss()
        self.optimizer = None


        self.LOG_FILE = os.path.join(RUNS_DIR, f'{hyperparameters_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{hyperparameters_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{hyperparameters_set}.png')

    def run(self,is_training=True,render=False):

        if is_training:
            log_message = f"DQN\n"

            if self.enable_double_dqn == True:
                log_message = f"Double DQN Enable"
            print(log_message)
            with open(self.LOG_FILE, 'w') as log_file:
                log_file.write(log_message+ '\n')
            start_time = datetime.datetime.now()
            last_graph_update_time = start_time
            log_message = f"Start time: {start_time}"
            print(log_message)
            with open(self.LOG_FILE, 'w') as log_file:
                log_file.write(log_message+ '\n')

        # env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
        env = gymnasium.make(self.env_id, render_mode='human' if render else None, **self.env_make_params)
        

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        reward_per_episode = []
        epsilon_history = []
        optimize_every_n_steps = 10
        policy_dqn = DQN(num_states,num_actions,self.fc1_nodes,self.enable_dueling_dqn ).to(device)

        if is_training:
            if self.pretrained_model and os.path.exists(self.pretrained_model):

                print(f"Loading pretrained model from {self.pretrained_model}")
                policy_dqn.load_state_dict(torch.load(self.pretrained_model))
                target_dqn.load_state_dict(policy_dqn.state_dict())
                print(f"Loaded pretrained model from {self.pretrained_model}")
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init
            
            target_dqn = DQN(num_states,num_actions,self.fc1_nodes,self.enable_dueling_dqn  ).to(device)
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
            state = torch.as_tensor(state,dtype=torch.float,device=device)

            terminated = False
            episode_reward = 0.0

            while (not terminated and episode_reward < self.stop_on_reward):
                # Next action:
                # (feed the observation to your agent here)
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.as_tensor(action,dtype=torch.long,device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Processing:
                new_state, reward, terminated, _, info = env.step(action.item())
                
                episode_reward += reward

                new_state = torch.as_tensor(new_state,dtype=torch.float,device=device)
                reward = torch.as_tensor(reward,dtype=torch.float,device=device)
                

                if is_training:
                    memory.append((state.detach(),action.detach() if action.requires_grad else action,new_state.detach(),reward.detach(),terminated))
                    step_count += 1
                    if step_count % optimize_every_n_steps == 0:
                        if len(memory) > self.mini_batch_size:

                            #sample from memory
                            mini_batch = memory.sample(self.mini_batch_size)
                            self.optimize(mini_batch,policy_dqn,target_dqn)

                            
                            if step_count > self.network_sync_rate:
                                target_dqn.load_state_dict(policy_dqn.state_dict())
                                step_count = 0
                state = new_state

            reward_per_episode.append(episode_reward)

            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.datetime.now().strftime(DATE_FORMAT)}: New best reward: {episode_reward} at episode {episode}"
                    print(f"Memory used: {psutil.virtual_memory().used / (1024 ** 3):.2f} GB")
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as log_file:
                        log_file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                current_time = datetime.datetime.now()
                if current_time - last_graph_update_time > datetime.timedelta(seconds=60):
                    self.save_graph(reward_per_episode, epsilon_history)
                    last_graph_update_time = current_time
                if len(memory) > self.mini_batch_size:
                                            #sample from memory
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch,policy_dqn,target_dqn)
                    epsilon = max(epsilon * self.epsilon_decay,self.epsilon_min)
                    epsilon_history.append(epsilon)
                    
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0
                    
            env.close() 
            # if epsilon <= self.epsilon_min:
            #     log_message = f"{datetime.datetime.now().strftime(DATE_FORMAT)}: Epsilon decay finished at episode {episode}"
            #     print(log_message)
            #     with open(self.LOG_FILE, 'a') as log_file:
            #         log_file.write(log_message + '\n')
            #     break


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
        plt.close()


    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states,actions,new_states,rewards,terminations = zip(*mini_batch)

        states = torch.stack(states).to(device)
        actions = torch.stack(actions).to(device)   
        new_states = torch.stack(new_states).to(device) 
        rewards = torch.stack(rewards).to(device)   
        terminations = torch.as_tensor(terminations).float().to(device)
        with torch.no_grad():

            if self.enable_double_dqn:
                best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)
                target_q = rewards + (1-terminations )* self.discount_factor_g  * target_dqn(new_states).gather(dim=1,index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
            else:
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