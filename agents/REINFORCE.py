import shutil
import sys
from collections import deque
from os.path import exists

import cv2
from agents.AutoEncoder import Encoder, Decoder
from agents.PolicyValueNetwork import PolicyValueNetwork
import torch
import numpy as np
from numpy.random import default_rng
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt

from agents.ReplayMemory import ReplayMemory
from environments.QuestEnvironment import QuestEnvironment


sys.path.append('../')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_filepath = 'model_weights.pth'
AE_encoder_model_filepath = 'E_model_weights.pth'
AE_decoder_model_filepath = 'D_model_weights.pth'
memory_replay_filepath = 'replay_buffer.pkl'
FULL_OBS_SPACE = 'glyphs'
OBS_SPACE = 'glyphs_crop'
AE_OBS_SPACE = 'pixel_crop'

# Here we define different maximium steps and different maze environments, depending on the difficulty setting
# This is used to train the agent on progressively more challenging mazes

max_steps_per_level = [2000, 2000, 5000, 5000, 10000, 10000, 15000, 15000]
mazes = ['MiniHack-MazeWalk-9x9-v0', 'MiniHack-MazeWalk-9x9-v0',
         'MiniHack-MazeWalk-15x15-v0', 'MiniHack-MazeWalk-15x15-v0',
         'MiniHack-MazeWalk-45x19-v0', 'MiniHack-MazeWalk-45x19-v0',
         'MiniHack-Quest-Hard-v0', 'MiniHack-Quest-Hard-v0']

# Each type of maze has 2 versions, one randomized and one seeded
seeds = [[75], None, [75, 373], None, [75, 89, 373], None, [75, 89, 373], None]

reward_win = 1000
rewards_maze_escape = [81, 81, 225, 225, 500, 500, 500, 500]
reward_lose = -20
step_penalty = -0.001
wall_penalty = -0.1

# The last n frames are saved, and passed as input to the agent

frame_stack_size = 4

# This function is used to approximate the 'true' value of each state, to compute the TD error
def compute_returns_naive_baseline(rewards, discount):

    baseline = sum(rewards) / len(rewards)
    R = sum(r*(discount**i) for i, r in enumerate(rewards)) - baseline
    return R

# This function loads in the appropriate environment and models, depending on the difficulty setting
def load_env_and_models(difficulty_level, agent_p_lr, agent_v_lr, AE_lr):

    environment = QuestEnvironment().create(
        env_type=mazes[difficulty_level],
        reward_lose=reward_lose,
        reward_win=reward_win,
        penalty_step=wall_penalty,
        penalty_time=step_penalty,
        max_episode_steps=max_steps_per_level[difficulty_level],
        seeds=seeds[difficulty_level]
    )

    # The latent space of the autoencoder - this is passed to the REINFORCE agent
    latent_space_dim = frame_stack_size*np.prod(environment.observation_space.spaces[OBS_SPACE].shape)

    # One-hot coordinate encodings, to represent the agent location
    one_hot_coords_length = np.prod(environment.observation_space.spaces[FULL_OBS_SPACE].shape)

    agent = PolicyValueNetwork(environment.action_space.n, latent_space_dim+2*one_hot_coords_length).to(device)

    # If we are loading from saved weights, reduce the learning rate
    if exists(model_filepath):
        agent_p_lr /= 10
        agent_v_lr /= 10
        agent.load_state_dict(torch.load(model_filepath))

    # Define optimizers and schedulers for our value and policy network

    agent_policy_optimizer = torch.optim.RMSprop(agent.policy_net.parameters(), lr=agent_p_lr)
    agent_policy_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(agent_policy_optimizer)
    agent_value_optimizer = torch.optim.RMSprop(agent.value_net.parameters(), lr=agent_v_lr)
    agent_value_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(agent_value_optimizer)

    # Calculate the input dimensions for the autoencoder
    AE_input_shape_dim = environment.observation_space.spaces[AE_OBS_SPACE].shape

    # Reshape if needed
    if len(AE_input_shape_dim) == 2:
        AE_input_shape_dim = (AE_input_shape_dim[0], AE_input_shape_dim[1], 1)

    AE_encoder = Encoder(batch_size=frame_stack_size, encoded_space_dim=latent_space_dim, image_size=frame_stack_size*AE_input_shape_dim).to(device)
    conv_shape = AE_encoder.get_convolution_shape()
    AE_decoder = Decoder(encoded_space_dim=latent_space_dim, image_size=AE_input_shape_dim, convolution_shape=conv_shape).to(device)

    # If we are loading from saved weights, reduce the learning rate
    if exists(AE_encoder_model_filepath):
        AE_encoder.load_state_dict(torch.load(AE_encoder_model_filepath))
        AE_lr /= 10

    if exists(AE_decoder_model_filepath):
        AE_decoder.load_state_dict(torch.load(AE_decoder_model_filepath))

    # Define optimizers and schedulers for our autoencoder
    AE_optim = torch.optim.Adam([{'params': AE_encoder.parameters()}, {'params': AE_decoder.parameters()}], lr=AE_lr,
                                weight_decay=1e-04)

    AE_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(AE_optim)

    return environment, max_steps_per_level[difficulty_level], agent, agent_policy_optimizer, agent_policy_sched, agent_value_optimizer, agent_value_sched, AE_encoder, AE_decoder, AE_optim, AE_sched, AE_input_shape_dim


if __name__ == "__main__":


    # Hyperparameters
    num_episodes = 10000
    agent_policy_alpha = 10e-4
    agent_value_alpha = 10e-3
    AE_alpha = 10e-5
    gamma = 0.995

    # Whether or not to trigger an episode end when the maze is reached - this is used to improve early training of the
    # agent to solve the first problem
    early_maze_break = False

    # Setting the starting difficulty for training
    maze_difficulty_level = 0
    max_difficulty_level = 7

    # Load in environment and models

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

    # Keep track of the average returns and loss across episodes
    return_means = []
    losses_means = []

    # We will record the last n returns, to decide whether to increase the difficulty level
    returns_history_length = 10
    returns = deque([], maxlen=returns_history_length)

    # Set a threshold to decide whether the autoencoder was accurate enough for the REINFORCE agent to be trained
    AE_threshold = 0.05

    # Total length and batch length settings for the autoencoder random replay buffer
    AE_memory_size = 1000
    AE_batch_size = 100
    AE_shuffle_factor = 0.1
    memory = ReplayMemory(AE_memory_size)

    rng = default_rng()

    # Whether to use the autoencoder, or the raw state
    use_AE = False

    # Begin generating episodes
    for k in range(num_episodes):
        obs = env.reset()

        # Initialize dictionaries for tracking state visits and rewards
        visit_counts = dict()
        visit_rewards = dict()

        # Initialize lists for tracking trajectory information
        Actions, Probs, Values, Rewards = [], [], [], []
        total_AE_loss = []

        # Default initial values
        done = False
        reward = -1
        prev_coords = (-1, -1)
        back_step_counter = 0
        wall_step_counter = 0
        escaped_maze = False
        reward_maze_escape = rewards_maze_escape[maze_difficulty_level]

        # Success for the first 6 levels is simply escaping the maze

        if maze_difficulty_level < 6:
            reward_success = reward_maze_escape
        else:
            reward_success = reward_win

        # Whether to use the autoencoder, or the raw state features

        if use_AE:
            AE_loss_list = []
            empty_frame = np.zeros((AE_input_shape[2], AE_input_shape[0], AE_input_shape[1]))
            frame_stack = deque([empty_frame] * frame_stack_size, maxlen=frame_stack_size)
        else:
            empty_state = np.zeros(obs[OBS_SPACE].shape)
            state_stack = deque([empty_state] * frame_stack_size, maxlen=frame_stack_size)

        # Matrix of state visits
        one_hot_visits = np.zeros(obs[FULL_OBS_SPACE].shape)
        one_hot_visits[int(obs['blstats'][1]), int(obs['blstats'][0])] = 1

        # Track frames of trajectory, for video
        run_frames = []

        # Begin generating trajectory
        for h in range(max_steps):

            # Track state visit
            one_hot_coords = np.zeros(obs[FULL_OBS_SPACE].shape)
            one_hot_coords[int(obs['blstats'][1]), int(obs['blstats'][0])] = 1

            # Add video frame
            run_frames.append(obs['pixel'])

            # Below, if the autoencoder is being used, the state pixels are passed into the decoder and the latent
            # space used for value and action prediction

            # If the autoencoder is not being used, the state glyphs are used for value and action prediction
            if use_AE:

                state_pixels = np.array(obs[AE_OBS_SPACE] / 255., dtype=np.float32).reshape(
                    (AE_input_shape[2], AE_input_shape[0], AE_input_shape[1]))

                frame_stack.append(state_pixels)
                feature_tensor = torch.tensor(np.array(frame_stack)).float().to(device)

                # Once the replay buffer has been suitably populated, only add 10% of states
                if len(memory) < AE_memory_size or rng.random() < AE_shuffle_factor:
                    memory.push(feature_tensor)

                # Retrieve latent representation of state
                encoded_state = encoder(feature_tensor)

                # Decode latent representation for comparison to true state
                decoded_state = decoder(encoded_state.to(device))

                decoded_frame = decoded_state.detach().clone().cpu().numpy().reshape(AE_input_shape)

                AE_loss = F.mse_loss(decoded_state.squeeze(0), torch.tensor(state_pixels).to(device))
                AE_optimizer.zero_grad()
                AE_loss.backward()
                AE_optimizer.step()

                AE_loss_list.append(AE_loss.item())

                combined = np.concatenate([decoded_frame, obs[AE_OBS_SPACE]/255.])

                cv2.imshow('Frame', combined)
                cv2.waitKey(1)

                state = torch.nn.functional.normalize(encoded_state.unsqueeze(0))

            else:
                state_stack.append(obs[OBS_SPACE])
                stack_tensor = torch.tensor(np.array(state_stack).flatten()).float().to(device)
                state = torch.nn.functional.normalize(stack_tensor.unsqueeze(0))

            # Append the state visit matrix, and current location to the state information

            state = torch.cat((state,
                               torch.tensor(one_hot_visits.flatten()).float().to(device).unsqueeze(0),
                               torch.tensor(one_hot_coords.flatten()).float().to(device).unsqueeze(0)), 1)

            # Sanity check
            if escaped_maze and early_maze_break:
                print('This shouldn\'t be possible')
                action, value, probs = 0, 0, 0
            else:
                # Predict action and state value
                action, probs = nn.action(state)
                value = nn.value(state)

            # Save trajectory information

            Actions.append(action)
            Values.append(value)
            Probs.append(probs)

            # Take action, and receive reward and observation

            obs_, r, done, info = env.step(action)
            env.render()

            # Check for maze escape

            coords = (int(obs_['blstats'][0]), int(obs_['blstats'][1]))

            one_hot_visits[coords[1], coords[0]] += 1

            # Modify reward according to position on map
            # Done with level
            if done:
                status = str(info['end_status'])
                if 'DEATH' in status or 'ABORTED' in status:
                    reward = reward_lose
                elif 'SUCCESS' in status:
                    reward = reward_success
                else:
                    print('This shouldn\'t be possible')
            # Escaped maze
            elif coords[0] > 27 and 8 < coords[1] < 14 and maze_difficulty_level >= 6:
                escaped_maze = True
                reward = reward_maze_escape

            # Still in maze
            else:
                if coords not in visit_counts:
                    visit_counts[coords] = 1
                    visit_rewards[coords] = r
                    reward = 1

                else:
                    visit_counts[coords] += 1
                    visit_rewards[coords] += r

                    if coords == prev_coords:
                        wall_step_counter += 1

                    elif coords[0] == prev_coords[0] or coords[1] == prev_coords[1]:
                        back_step_counter += 1
                    # Sanity check
                    else:
                        print('This shouldn\'t be possible')

                    #reward = r + np.e*np.sqrt(np.log(visit_counts[coords])/visit_counts[coords]) + visit_rewards[coords]/visit_counts[coords]

                    # The agent is penalized for non-exploration

                    non_exploration_penalty = np.log(visit_counts[coords]) / np.sqrt(visit_counts[coords])
                    reward = (visit_rewards[coords]/visit_counts[coords]) * non_exploration_penalty

            print(f"\rEpisode: {k + 1}, Step: {h + 1}, Action: {action}, Reward: {reward}", end="")

            Rewards.append(reward)
            prev_coords = coords
            obs = obs_

            if done or (escaped_maze and early_maze_break):
                break

        # Auto-encoder training

        if use_AE and len(memory) >= AE_batch_size:
            states = memory.sample(AE_batch_size)
            for state_pixels in states:
                encoded_state = encoder(state_pixels)
                decoded_state = decoder(encoded_state.to(device))

                AE_loss = F.mse_loss(decoded_state.squeeze(0), state_pixels[-1])
                AE_loss_list.append(AE_loss.item())
                AE_optimizer.zero_grad()
                AE_loss.backward()
                AE_optimizer.step()

        returns.append(reward)

        T = len(Rewards)

        Values = torch.tensor(Values, requires_grad=True).float().to(device)
        Probs = torch.tensor(Probs, requires_grad=True).float().to(device)

        # True baseline estimation

        with torch.no_grad():

            G = [compute_returns_naive_baseline(Rewards[t:], gamma) for t in range(T)]
            G = torch.tensor(G, requires_grad=False).float().to(device)
            G = (G - G.mean()) / G.std()
            # TD error calculation
            errors = G - Values

        # Policy gradient
        policy_loss = - errors * Probs
        cum_policy_loss = policy_loss.sum()
        # cum_value_loss = F.mse_loss(Values, G).to(device)

        # Value gradient
        value_loss = - errors * Values
        cum_value_loss = value_loss.sum()

        # If using the autoencoder, the agent is only trained if the decoded representation is accurate enough

        if not use_AE or np.mean(AE_loss_list) < AE_threshold:

            policy_optimizer.zero_grad()
            cum_policy_loss.backward(retain_graph=True)
            policy_optimizer.step()

            value_optimizer.zero_grad()
            cum_value_loss.backward()
            value_optimizer.step()

            agent_policy_scheduler.step(cum_policy_loss)
            agent_value_scheduler.step(cum_value_loss)
        else:
            print('\nLatent representation not accurate enough to facilitate training.')

        v_loss_list = np.abs(errors.detach().cpu().numpy())
        p_loss_list = policy_loss.detach().cpu().numpy()

        Loss = v_loss_list + p_loss_list

        if use_AE:
            AE_print = f"\tAE Loss: {np.sum(AE_loss_list):0.4f} ({np.mean(AE_loss_list):0.4f})"
        else:
            AE_print = ""

        # Print episode statistics

        print(f"\n\nLevel: {maze_difficulty_level+1}/{max_difficulty_level+1} "
              f"\tRewards: {np.sum(Rewards):0.4f} ({np.mean(Rewards):0.4f}), States Visited: {len(visit_counts)}"
              f"\nPolicy Loss: {np.sum(p_loss_list):0.4f} ({np.mean(p_loss_list):0.4f})"
              f"\tValue Loss: {np.sum(v_loss_list):0.4f} ({np.mean(v_loss_list):0.4f})"
              f"{AE_print}"
              f"\nNew states: {len(visit_counts)} / {h + 1} ({round(100*len(visit_counts)/(h + 1), 2)}%) "
              f"\tBack-step states: {back_step_counter} / {h + 1} ({round(100*back_step_counter/(h + 1), 2)}%) "
              f"\tWall-hits: {wall_step_counter} / {h + 1} ({round(100*wall_step_counter/(h + 1), 2)}%)\n")

        # Backup previous weights in case of corruption

        if exists(model_filepath):
            shutil.copy2(model_filepath, 'backups/')

        if exists(AE_encoder_model_filepath):
            shutil.copy2(AE_encoder_model_filepath, 'backups/')

        if exists(AE_decoder_model_filepath):
            shutil.copy2(AE_decoder_model_filepath, 'backups/')

        # Save new agent weights

        torch.save(nn.state_dict(), model_filepath)
        torch.save(encoder.state_dict(), AE_encoder_model_filepath)
        torch.save(decoder.state_dict(), AE_decoder_model_filepath)

        # Increase difficulty if the agent has achieved success in 85% of the recent runs

        if len(returns) >= returns_history_length and np.mean(returns)/reward_success > 0.85:
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
            returns = deque([], maxlen=returns_history_length)
            k = 0

        # Plot graphs

        return_means.append(np.mean(Rewards))
        losses_means.append(np.mean(Loss))

        plt.ioff()
        window = int(max_steps)
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.title('Return')
        plt.xlabel('Episodes')
        plt.ylabel('Return')
        rolled_returns = pd.DataFrame(return_means).rolling(min(20, len(return_means))).mean()[0]
        std_returns = pd.DataFrame(return_means).rolling(min(20, len(return_means))).std()[0]
        plt.plot(rolled_returns)
        x = np.arange(len(rolled_returns))
        plt.fill_between(x, rolled_returns + std_returns, rolled_returns - std_returns, facecolor='blue', alpha=0.5)
        plt.grid(True)
        plt.subplot(122)
        plt.title('Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.plot(pd.DataFrame(losses_means))
        plt.plot(pd.DataFrame(losses_means).rolling(min(20, len(losses_means))).mean())
        plt.grid(True)
        plt.savefig('REINFORCE.png')
        plt.clf()
        plt.close()

        # Save video of trajectory

        video_name = 'maze_escape.mp4' if escaped_maze else 'animation.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_name, fourcc, 30, (run_frames[0].shape[1], run_frames[0].shape[0]))
        for frame in run_frames:
            video.write(frame[:, :, ::-1])
        video.release()

    cv2.destroyAllWindows()
    env.close()
