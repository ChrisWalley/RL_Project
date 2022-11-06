import pickle
import shutil
import sys
from collections import deque
from os.path import exists
import cv2
from agents.AutoEncoder import Encoder, Decoder
from agents.PolicyValueNetwork import PolicyValueNetwork
import math
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt

from agents.ReplayMemory import ReplayMemory
from environments.QuestEnvironment import QuestEnvironment

# resolve path for notebook
sys.path.append('../')
# if there is a Cuda GPU, then we want to use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_filepath = 'model_weights.pth'
AE_encoder_model_filepath = 'E_model_weights.pth'
AE_decoder_model_filepath = 'D_model_weights.pth'
memory_replay_filepath = 'replay_buffer.pkl'
OBS_SPACE = 'glyphs_crop'
AE_OBS_SPACE = 'pixel_crop'
max_steps_per_level = [2000, 5000, 10000, 15000]
mazes = ['MiniHack-MazeWalk-9x9-v0', 'MiniHack-MazeWalk-15x15-v0', 'MiniHack-MazeWalk-45x19-v0', 'MiniHack-Quest-Hard-v0']

reward_lose = -0.5
reward_win = 50
reward_escape_maze = 10
reward_exploration = 0.01
step_penalty = -0.0001
wall_penalty = -0.001

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


def compute_returns_naive_baseline(rewards, gamma):

    baseline = sum(rewards) / len(rewards)
    returns = sum(r*(gamma**i) for i, r in enumerate(rewards)) - baseline
    return returns


def load_env_and_models(difficulty_level, agent_p_lr, agent_v_lr, AE_lr):

    environment = QuestEnvironment().create(
        env_type=mazes[difficulty_level],
        reward_lose=reward_lose,
        reward_win=reward_win,
        penalty_step=wall_penalty,
        penalty_time=step_penalty,
        max_episode_steps=max_steps_per_level[difficulty_level],
        #seed=seed
    )
    latent_space_dim = 3*environment.observation_space.spaces[OBS_SPACE].shape[0] *\
                       environment.observation_space.spaces[OBS_SPACE].shape[1]

    agent = PolicyValueNetwork(environment.action_space.n, latent_space_dim).to(device)

    if exists(model_filepath):
        agent_p_lr /= 10
        agent_v_lr /= 10
        agent.load_state_dict(torch.load(model_filepath))

    agent_policy_optimizer = torch.optim.RMSprop(agent.policy_net.parameters(), lr=agent_p_lr)
    agent_policy_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(agent_policy_optimizer)
    agent_value_optimizer = torch.optim.RMSprop(agent.value_net.parameters(), lr=agent_v_lr)
    agent_value_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(agent_value_optimizer)

    AE_input_shape_dim = environment.observation_space.spaces[AE_OBS_SPACE].shape

    if len(AE_input_shape_dim) == 2:
        AE_input_shape_dim = (AE_input_shape_dim[0], AE_input_shape_dim[1], 1)

    AE_encoder = Encoder(encoded_space_dim=latent_space_dim, fc2_input_dim=AE_input_shape_dim).to(device)
    AE_decoder = Decoder(encoded_space_dim=latent_space_dim, fc2_input_dim=AE_input_shape_dim).to(device)

    if exists(AE_encoder_model_filepath):
        AE_encoder.load_state_dict(torch.load(AE_encoder_model_filepath))
        AE_lr /= 10

    if exists(AE_decoder_model_filepath):
        AE_decoder.load_state_dict(torch.load(AE_decoder_model_filepath))

    AE_optim = torch.optim.Adam([{'params': AE_encoder.parameters()}, {'params': AE_decoder.parameters()}], lr=AE_lr,
                                weight_decay=1e-04)

    AE_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(AE_optim)

    return environment, max_steps_per_level[difficulty_level], agent, agent_policy_optimizer, agent_policy_sched, agent_value_optimizer, agent_value_sched, AE_encoder, AE_decoder, AE_optim, AE_sched, AE_input_shape_dim


if __name__ == "__main__":

    num_episodes = 1000
    agent_policy_alpha = 10e-5
    agent_value_alpha = 10e-5
    AE_alpha = 10e-5
    gamma = 0.9995
    seed = 42
    early_maze_break = True

    maze_difficulty_level = 0
    max_difficulty_level = 3

    variables_and_networks = load_env_and_models(maze_difficulty_level, agent_policy_alpha, agent_value_alpha, AE_alpha)

    env = variables_and_networks[0]
    max_steps = variables_and_networks[1]
    nn = variables_and_networks[2]
    policy_optimizer = variables_and_networks[3]
    agent_policy_scheduler = variables_and_networks[4]
    value_optimizer = variables_and_networks[5]
    agent_value_scheduler = variables_and_networks[6]
    encoder = variables_and_networks[7]
    decoder = variables_and_networks[8]
    AE_optimizer = variables_and_networks[9]
    AE_scheduler = variables_and_networks[10]
    AE_input_shape = variables_and_networks[11]

    # training
    cumulative_returns = []
    cumulative_losses = []

    returns_history_length = 10

    AE_threshold = 0.05
    AE_memory_size = 10000
    AE_batch_size = 100

    returns = deque([], maxlen=returns_history_length)

    memory = ReplayMemory(AE_memory_size)
    if exists(memory_replay_filepath):
        memory.load(memory_replay_filepath)

    for k in range(num_episodes):
        obs = env.reset()

        visit_counts = dict()
        coord_rewards = dict()

        done = False
        Actions, States, Probs, Values, Rewards = [], [], [], [], []
        total_AE_loss = []
        reward = -10

        prev_coords = (-1, -1)

        back_step_counter = 0
        wall_step_counter = 0
        new_state_counter = 0
        escaped_maze = False

        for h in range(max_steps):

            state_pixels = np.array(obs[AE_OBS_SPACE] / 255., dtype=np.float32).reshape(
                (AE_input_shape[2], AE_input_shape[0], AE_input_shape[1]))

            memory.push(state_pixels)

            encoded_state = encoder(state_pixels)

            decoded_state = decoder(encoded_state.to(device))

            decoded_frame = decoded_state.detach().clone().cpu().numpy().reshape(AE_input_shape)

            AE_loss = F.mse_loss(decoded_state.squeeze(0), torch.tensor(state_pixels).to(device))
            AE_optimizer.zero_grad()
            AE_loss.backward()
            AE_optimizer.step()

            combined = np.concatenate([decoded_frame, obs[AE_OBS_SPACE]/255.])

            cv2.imshow('Frame', combined)
            cv2.waitKey(1)

            States.append(state_pixels)

            if escaped_maze:
                print('This shouldn\'t be possible')
                action, value, probs = 0, 0, 0
            else:
                action, probs = nn.action(encoded_state)
                value = nn.value(encoded_state)

            Actions.append(action)
            Values.append(value)
            Probs.append(probs)

            obs_, r, done, info = env.step(action)
            env.render()

            coords = (int(obs_['blstats'][0]), int(obs_['blstats'][1]))

            # Modify reward according to position on map
            # Done with leve
            if done:
                status = str(info['end_status'])
                if 'DEATH' in status:
                    reward = reward_lose
                elif 'SUCCESS' in status:
                    reward = reward_win
                else:
                    print('This shouldn\'t be possible')
            # Escaped maze
            elif coords[0] > 29 and 8 < coords[1] < 14 and maze_difficulty_level < 3:
                escaped_maze = True
                reward = reward_escape_maze

            # Still in maze
            else:
                # Include action?
                if coords not in visit_counts:
                    visit_counts[coords] = 1
                    coord_rewards[coords] = reward

                    new_state_counter += 1

                else:
                    visit_counts[coords] += 1
                    coord_rewards[coords] += reward

                    if coords == prev_coords:
                        wall_step_counter += 1
                    elif coords[0] == prev_coords[0] or coords[1] == prev_coords[1]:
                        back_step_counter += 1
                    else:
                        print('This shouldn\'t be possible')

                non_exploration_penalty = math.sqrt(visit_counts[coords] / (2*math.log(h+2)))
                reward = r * non_exploration_penalty

            print(f"\rEpisode: {k + 1}, Step: {h + 1}, Action: {action}, Reward: {reward}", end="")

            Rewards.append(reward)
            prev_coords = coords
            obs = obs_

            if done or (escaped_maze and early_maze_break):
                break

        if len(memory) < AE_batch_size:
            continue
        states = memory.sample(AE_batch_size)
        AE_loss_list = []
        encoder.train()
        decoder.train()
        for state_pixels in states:
            encoded_state = encoder(state_pixels)
            decoded_state = decoder(encoded_state.to(device))

            AE_loss = F.mse_loss(decoded_state.squeeze(0), torch.tensor(state_pixels).to(device))
            AE_loss_list.append(AE_loss.item())
            AE_optimizer.zero_grad()
            AE_loss.backward()
            AE_optimizer.step()

        returns.append(reward)

        T = len(Rewards)

        G = [compute_returns_naive_baseline(Rewards[t:], gamma) for t in range(T)]
        G = torch.tensor(G, requires_grad=True).float().to(device)
        G = (G - G.mean()) / G.std()

        Values = torch.tensor(Values, requires_grad=True).float().to(device)
        Probs = torch.tensor(Probs, requires_grad=True).float().to(device)

        #errors = Values - G
        errors = G - Values
        policy_loss = - errors * Probs
        value_loss = F.mse_loss(Values, G).to(device)

        #AE_scheduler.step(AE_losses.sum())
        #AE_loss_list = AE_losses.detach().cpu().numpy()

        if np.mean(AE_loss_list) < AE_threshold:

            policy_optimizer.zero_grad()
            policy_loss.sum().backward(retain_graph=True)
            policy_optimizer.step()

            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()

            agent_policy_scheduler.step(policy_loss.sum())
            agent_value_scheduler.step(value_loss)
        else:
            print('Latent representation not accurate enough to facilitate training.')

        v_loss_list = np.abs(errors.detach().cpu().numpy())
        p_loss_list = policy_loss.detach().cpu().numpy()

        Loss = v_loss_list + p_loss_list

        print(f"\n\nRewards: {np.sum(Rewards):0.4f} ({np.mean(Rewards):0.4f}), States Visited: {len(visit_counts)}"
              f"\nPolicy Loss: {np.sum(p_loss_list):0.4f} ({np.mean(p_loss_list):0.4f})"
              f"\tValue Loss: {np.sum(v_loss_list):0.4f} ({np.mean(v_loss_list):0.4f})"
              f"\tAE Loss: {np.sum(AE_loss_list):0.4f} ({np.mean(AE_loss_list):0.4f})"
              f"\nNew states: {new_state_counter} / {h + 1} ({round(100*new_state_counter/(h + 1), 2)}%) "
              f"\tBack-step states: {back_step_counter} / {h + 1} ({round(100*back_step_counter/(h + 1), 2)}%) "
              f"\tWall-hits: {wall_step_counter} / {h + 1} ({round(100*wall_step_counter/(h + 1), 2)}%)\n")

        if exists(model_filepath):
            shutil.copy2(model_filepath, 'backups/')

        if exists(AE_encoder_model_filepath):
            shutil.copy2(AE_encoder_model_filepath, 'backups/')

        if exists(AE_decoder_model_filepath):
            shutil.copy2(AE_decoder_model_filepath, 'backups/')

        torch.save(nn.state_dict(), model_filepath)
        torch.save(encoder.state_dict(), AE_encoder_model_filepath)
        torch.save(decoder.state_dict(), AE_decoder_model_filepath)

        memory.save(memory_replay_filepath)

        if len(returns) > returns_history_length and np.mean(returns)/reward_win > 0.85:
            print('\n\n Increasing environment difficulty! \n\n')
            maze_difficulty_level = min(maze_difficulty_level+1, max_difficulty_level)
            env.close()
            variables_and_networks = load_env_and_models(maze_difficulty_level, agent_policy_alpha, agent_value_alpha,
                                                         AE_alpha)

            env = variables_and_networks[0]
            max_steps = variables_and_networks[1]
            nn = variables_and_networks[2]
            policy_optimizer = variables_and_networks[3]
            agent_policy_scheduler = variables_and_networks[4]
            value_optimizer = variables_and_networks[5]
            agent_value_scheduler = variables_and_networks[6]
            encoder = variables_and_networks[7]
            decoder = variables_and_networks[8]
            AE_optimizer = variables_and_networks[9]
            AE_scheduler = variables_and_networks[10]
            AE_input_shape = variables_and_networks[11]

            env.reset()
            k = 0

        Loss = np.reshape(Loss, (np.shape(Loss)[0], 1))

        cumulative_returns.append(sum(Rewards))
        cumulative_losses.append(sum(Loss))
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
        plt.plot(pd.DataFrame(cumulative_losses))
        plt.plot(pd.DataFrame(cumulative_losses).rolling(min(10, len(cumulative_losses))).mean())
        plt.grid(True)
        plt.savefig('REINFORCE.png')
        plt.close()

    cv2.destroyAllWindows()
    env.close()
