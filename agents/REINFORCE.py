import sys
from os.path import exists

from agents.AutoEncoder import Encoder, Decoder
from agents.PolicyValueNetwork import PolicyValueNetwork

# resolve path for notebook
sys.path.append('../')
import math
import torch
import random
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from environments.QuestEnvironment import QuestEnvironment

# if there is a Cuda GPU, then we want to use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_filepath = 'model_weights.pth'
AE_encoder_model_filepath = 'E_model_weights.pth'
AE_decoder_model_filepath = 'D_model_weights.pth'
OBS_SPACE = 'glyphs'
AE_OBS_SPACE = 'pixel'
max_steps_per_level = [500, 2000, 5000, 5000]
mazes = ['MiniHack-MazeWalk-9x9-v0', 'MiniHack-MazeWalk-15x15-v0', 'MiniHack-MazeWalk-45x19-v0', 'MiniHack-Quest-Hard-v0']

def get_exploration_reward(state, reward):
    s = state['blstats']
    coords = (int(s[0]), int(s[1]))

    if coords not in visit_counts:
        visit_counts[coords] = 1
        coord_rewards[coords] = reward
        return 0.01
    else:
        visit_counts[coords] += 1
        coord_rewards[coords] += reward

    return reward * math.sqrt(math.log(visit_counts[coords]))

def convert_observation(obs, space, epsilon=10e-3):
    obs = obs[space]
    obs = torch.tensor(obs, dtype=torch.float)
    obs = torch.flatten(obs)
    obs = torch.reshape(obs, (1, obs.shape[0]))
    obs = torch.nn.functional.normalize(obs, p=2.0, dim=1, eps=epsilon, out=None)
    return obs


def compute_returns(rewards, gamma):
    returns = sum(r*(gamma**i) for i, r in enumerate(rewards))
    return returns


def load_env_and_models(difficulty_level, agent_lr, AE_lr):

    environment = QuestEnvironment().create(
        env_type=mazes[difficulty_level],
        reward_lose=reward_lose,
        reward_win=reward_win,
        penalty_step=-0.001,
        penalty_time=-0.0001,
        max_episode_steps=max_steps_per_level[difficulty_level],
        seed=seed
    )
    latent_space_dim = environment.observation_space.spaces[OBS_SPACE].shape[0] *\
                       environment.observation_space.spaces[OBS_SPACE].shape[1]

    agent = PolicyValueNetwork(environment.action_space.n, agent_lr, latent_space_dim).to(device)

    if exists(model_filepath):
        agent.load_state_dict(torch.load(model_filepath))

    AE_input_shape_dim = environment.observation_space.spaces[AE_OBS_SPACE].shape

    if len(AE_input_shape_dim) == 2:
        AE_input_shape_dim = (AE_input_shape_dim[0], AE_input_shape_dim[1], 1)

    AE_encoder = Encoder(encoded_space_dim=latent_space_dim, fc2_input_dim=AE_input_shape_dim).to(device)
    AE_decoder = Decoder(encoded_space_dim=latent_space_dim, fc2_input_dim=AE_input_shape_dim).to(device)

    AE_optim = torch.optim.Adam([{'params': AE_encoder.parameters()}, {'params': AE_decoder.parameters()}], lr=AE_lr,
                                weight_decay=1e-05)

    if exists(AE_encoder_model_filepath):
        AE_encoder.load_state_dict(torch.load(AE_encoder_model_filepath))

    if exists(AE_decoder_model_filepath):
        AE_decoder.load_state_dict(torch.load(AE_decoder_model_filepath))

    return environment, max_steps_per_level[difficulty_level], agent, AE_encoder, AE_decoder, AE_optim, AE_input_shape_dim


if __name__ == "__main__":

    num_episodes = 1000
    agent_alpha = 10e-5
    AE_alpha = 10e-5
    gamma = 0.995
    seed = 42
    reward_lose = -1
    reward_win = 1

    maze_difficulty_level = 0
    max_difficulty_level = 3

    env, max_steps, nn, encoder, decoder, AE_optimizer, AE_input_shape = load_env_and_models(maze_difficulty_level,
                                                                                             agent_alpha, AE_alpha)
    env.seed(seed)

    # training
    cumulative_returns = []

    for k in range(num_episodes):
        obs = env.reset()

        visit_counts = dict()
        coord_rewards = dict()

        done = False
        encoded_states, Actions, States, Probs, Values, Rewards = [], [], [], [], [], []
        total_AE_loss = []
        for h in range(max_steps):

            state_pixels = np.array(obs[AE_OBS_SPACE]/255., dtype=np.float32).reshape((AE_input_shape[2], AE_input_shape[0], AE_input_shape[1]))

            encoded_state = encoder(state_pixels)
            decoded_state = decoder(encoded_state.to(device))
            AE_loss = F.mse_loss(decoded_state, torch.tensor(state_pixels).to(device))

            AE_optimizer.zero_grad()
            AE_loss.backward()
            AE_optimizer.step()

            total_AE_loss.append(AE_loss.item())

            state = encoded_state

            action, probs = nn.action(state)
            value = nn.value(state)

            States.append(state_pixels)
            Actions.append(action)
            Values.append(value)
            encoded_states.append(encoded_state)
            Probs.append(probs)

            obs_, r, done, info = env.step(action)
            env.render()

            if done:
                status = str(info['end_status'])
                if 'DEATH' in status:
                    reward = reward_lose
                elif 'SUCCESS' in status:
                    reward = reward_win
                else:
                    reward = get_exploration_reward(obs_, r)
            else:
                reward = get_exploration_reward(obs_, r)

            print(f"\rEpisode: {k + 1}, Step: {h + 1}, Action: {action}, Reward: {reward}", end="")

            Rewards.append(reward)

            obs = obs_

            if done:
                break

        T = len(Rewards)

        G = [compute_returns(Rewards[t:], gamma) for t in range(T)]
        G = torch.tensor(G, requires_grad=True).float().to(device)
        G = (G - G.mean()) / G.std()

        Values = torch.tensor(Values, requires_grad=True).float().to(device)
        Probs = torch.tensor(Probs, requires_grad=True).float().to(device)

        errors = (G-Values)
        policy_loss = - Probs * errors
        nn.policy_optimizer.zero_grad()
        policy_loss.sum().backward(retain_graph=True)
        nn.policy_optimizer.step()

        value_loss = F.mse_loss(Values, G).to(device)
        nn.value_optimizer.zero_grad()
        value_loss.backward()
        nn.value_optimizer.step()
        v_loss_list = np.abs(errors.detach().cpu().numpy())
        p_loss_list = policy_loss.detach().cpu().numpy()

        Loss = v_loss_list + p_loss_list

        print(f"\n\nRewards: {np.sum(Rewards):0.4f} ({np.mean(Rewards):0.4f}), States Visited: {len(visit_counts)}"
              f"\nPolicy Loss: {np.sum(p_loss_list):0.4f} ({np.mean(p_loss_list):0.4f})"
              f"\tValue Loss: {np.sum(v_loss_list):0.4f} ({np.mean(v_loss_list):0.4f})"
              f"\tAE Loss: {np.sum(total_AE_loss):0.4f}, ({np.mean(total_AE_loss):0.4f})\n")

        torch.save(nn.state_dict(), model_filepath)
        torch.save(encoder.state_dict(), AE_encoder_model_filepath)
        torch.save(decoder.state_dict(), AE_decoder_model_filepath)


        if len(cumulative_returns) > 10 and np.mean(cumulative_returns[-10:])/reward_win > 0.85:
            print('\n\n Increasing environment difficulty! \n\n')
            maze_difficulty_level = min(maze_difficulty_level+1, max_difficulty_level)
            env.close()
            env, max_steps, nn, encoder, decoder, AE_optimizer, AE_input_shape = load_env_and_models(
                maze_difficulty_level, agent_alpha, AE_alpha)
            env.seed(seed)
            env.reset()
            k = 0

        Loss = np.reshape(Loss, (np.shape(Loss)[0], 1))

        cumulative_returns.append(sum(Rewards))
        # Plot stuff
        window = int(max_steps)
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.title('Return')
        plt.xlabel('Episodes')
        plt.ylabel('Return')
        plt.plot(pd.DataFrame(cumulative_returns))
        plt.plot(pd.DataFrame(cumulative_returns).rolling(min(10, len(cumulative_returns))).mean())
        plt.grid(True)
        plt.subplot(122)
        plt.title('Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.plot(pd.DataFrame(Loss))
        plt.plot(pd.DataFrame(Loss).rolling(10).mean())
        plt.grid(True)
        plt.savefig('REINFORCE.png')
        plt.close()
        #plt.show()

    for _ in range(5):

        Rewards = []

        obs = env.reset()
        obs = convert_observation(obs)

        done = False
        env.render()

        steps = 0

        while not done and steps <= max_steps:
            steps += 1

            probs = nn(obs)

            c = torch.distributions.Categorical(probs=probs)
            action = c.sample().item()

            obs_, rew, done, _info = env.step(action)
            obs_ = convert_observation(obs_)
            env.render()

            Rewards.append(rew)

        print(f'Reward: {sum(Rewards)}')

    env.close()