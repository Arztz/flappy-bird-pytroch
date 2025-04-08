import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
import torch
from experience_replay import ReplayMemory
import itertools
import yaml
import random

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
            epsilon_decay = self.epsilon_decay
            epsilon_min = self.epsilon_min
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

                state = new_state

            reward_per_episode.append(episode_reward)

            epsilon = max(epsilon * epsilon_decay,epsilon_min)
            epsilon_history.append(epsilon)


if __name__ == "__main__":
    agent = Agent('cartpole1')
    agent.run(is_training=True,render=True)
    # agent.run(is_training=False,render=True)