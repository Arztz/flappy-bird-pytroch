import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
import torch
from experience_replay import ReplayMemory
import itertools
import yaml
import random
import torch.nn as nn
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
        self.loss_fn = nn.MSELoss()
        self.optimizer = None


    def run(self,is_training=True,render=False):
        # env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)
        

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        reward_per_episode = []
        epsilon_history = []

        policy_dqn = DQN(num_states,num_actions ).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init
            
            target_dqn = DQN(num_states,num_actions ).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            step_count = 0

            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
        #  Training loop
        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state,dtype=torch.float,device=device)

            terminated = False
            episode_reward = 0.0

            while not terminated:
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

            epsilon = max(epsilon * self.epsilon_decay,self.epsilon_min)
            epsilon_history.append(epsilon)

            if len(memory) > self.mini_batch_size:

                #sample from memory
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch,policy_dqn,target_dqn)
                if step_count > self.network_sync_rate == 0:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        for state, action, new_state, reward, terminated in mini_batch:
            
            if terminated:
                target = reward
            else:
                with torch.no_grad():
                    target_q = reward + self.discount_factor_g * target_dqn(new_state).max()
            
            current_q = policy_dqn(state)

            loss = self.loss_fn(current_q,target_q)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
if __name__ == "__main__":
    agent = Agent('cartpole1')
    agent.run(is_training=True,render=True)
    # agent.run(is_training=False,render=True)