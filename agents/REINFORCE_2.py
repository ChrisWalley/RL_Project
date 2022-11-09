import pickle
import shutil
import sys
from collections import deque
from os.path import exists

import cv2
from agents.PolicyValueNetwork import PolicyValueNetwork
import math
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt

from agents.ReplayMemory import ReplayMemory
from environments.QuestEnvironment import QuestEnvironment


sys.path.append('../')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_filepath = 'model_2_weights.pth'
OBS_SPACE = 'glyphs_crop'

max_steps_per_level = [2000, 2000, 5000, 5000, 10000, 10000, 15000, 15000]
mazes = ['MiniHack-MazeWalk-9x9-v0', 'MiniHack-MazeWalk-9x9-v0',
         'MiniHack-MazeWalk-15x15-v0', 'MiniHack-MazeWalk-15x15-v0',
         'MiniHack-MazeWalk-45x19-v0', 'MiniHack-MazeWalk-45x19-v0',
         'MiniHack-Quest-Hard-v0', 'MiniHack-Quest-Hard-v0']

reward_win = 1000
rewards_maze_escape = [81, 81, 225, 225, 500, 500, 500, 500]
seeds = [[75], None, [75, 373], None, [75, 89, 373], None, [75, 89, 373], None]
reward_lose = -20
step_penalty = -0.001
wall_penalty = -0.1

frame_stack_size = 20


def compute_returns_naive_baseline(rewards, discount):

    baseline = sum(rewards) / len(rewards)
    R = sum(r*(discount**i) for i, r in enumerate(rewards)) - baseline
    return R


def frame_stack_to_tensor(stack):
    frames = []
    for frame in stack:
        frames.extend(frame)

    return torch.tensor(frames).float().to(device).unsqueeze(0)


def load_env_and_model(difficulty_level, agent_p_lr, agent_v_lr):

    environment = QuestEnvironment().create(
        env_type=mazes[difficulty_level],
        reward_lose=reward_lose,
        reward_win=reward_win,
        penalty_step=wall_penalty,
        penalty_time=step_penalty,
        max_episode_steps=max_steps_per_level[difficulty_level],
        seeds=seeds[difficulty_level]
    )
    latent_space_dim = frame_stack_size*np.prod(environment.observation_space.spaces[OBS_SPACE].shape)

    agent = PolicyValueNetwork(environment.action_space.n, latent_space_dim).to(device)

    if exists(model_filepath):
        agent_p_lr /= 10
        agent_v_lr /= 10
        agent.load_state_dict(torch.load(model_filepath))

    agent_policy_optimizer = torch.optim.RMSprop(agent.policy_net.parameters(), lr=agent_p_lr)
    agent_value_optimizer = torch.optim.RMSprop(agent.value_net.parameters(), lr=agent_v_lr)

    return environment, max_steps_per_level[difficulty_level], agent, agent_policy_optimizer, agent_value_optimizer


if __name__ == "__main__":

    num_episodes = 1000
    agent_policy_alpha = 10e-5
    agent_value_alpha = 10e-5
    gamma = 0.9995
    seed = 42
    early_maze_break = True

    maze_difficulty_level = 7
    max_difficulty_level = 7

    variables_and_networks = load_env_and_model(maze_difficulty_level, agent_policy_alpha, agent_value_alpha)

    env = variables_and_networks[0]
    max_steps = variables_and_networks[1]
    nn = variables_and_networks[2]
    policy_optimizer = variables_and_networks[3]
    value_optimizer = variables_and_networks[4]

    # training
    cumulative_returns = []
    cumulative_losses = []

    returns_history_length = 10

    returns = deque([], maxlen=returns_history_length)

    for k in range(num_episodes):
        obs = env.reset()

        visit_counts = dict()
        coord_rewards = dict()

        done = False
        Actions, States, Probs, Values, Rewards = [], [], [], [], []
        reward = -10

        prev_coords = (-1, -1)

        back_step_counter = 0
        wall_step_counter = 0
        new_state_counter = 0
        escaped_maze = False
        reward_maze_escape = rewards_maze_escape[maze_difficulty_level]

        if maze_difficulty_level < 6:
            reward_success = reward_maze_escape
        else:
            reward_success = reward_win

        empty_frame = np.zeros(obs[OBS_SPACE].flatten().shape)

        frame_stack = deque([empty_frame] * frame_stack_size, maxlen=frame_stack_size)

        for h in range(max_steps):

            frame_stack.append(obs[OBS_SPACE].flatten())

            feature_tensor = frame_stack_to_tensor(frame_stack)

            if escaped_maze:
                print('This shouldn\'t be possible')
                action, value, probs = 0, 0, 0
            else:
                action, probs = nn.action(feature_tensor)
                value = nn.value(feature_tensor)

            Actions.append(action)
            Values.append(value)
            Probs.append(probs)

            obs_, r, done, info = env.step(action)
            env.render()

            coords = (int(obs_['blstats'][0]), int(obs_['blstats'][1]))

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
            elif coords[0] > 29 and 8 < coords[1] < 14 and maze_difficulty_level >= 6:
                escaped_maze = True
                reward = reward_maze_escape

            # Still in maze
            else:
                # Include action?
                if coords not in visit_counts:
                    visit_counts[coords] = 1
                    new_state_counter += 1
                    reward = 1

                else:
                    visit_counts[coords] += 1
                    if coords == prev_coords:
                        wall_step_counter += 1
                    elif coords[0] == prev_coords[0] or coords[1] == prev_coords[1]:
                        back_step_counter += 1
                    else:
                        print('This shouldn\'t be possible')

                    reward = r

            print(f"\rEpisode: {k + 1}, Step: {h + 1}, Action: {action}, Reward: {reward}", end="")

            Rewards.append(reward)
            prev_coords = coords
            obs = obs_

            if done or (escaped_maze and early_maze_break):
                break

        returns.append(reward)

        T = len(Rewards)

        G = [compute_returns_naive_baseline(Rewards[t:], gamma) for t in range(T)]
        G = torch.tensor(G, requires_grad=True).float().to(device)
        G = (G - G.mean()) / G.std()

        Values = torch.tensor(Values, requires_grad=True).float().to(device)
        Probs = torch.tensor(Probs, requires_grad=True).float().to(device)

        errors = G - Values
        policy_loss = - errors * Probs
        value_loss = F.mse_loss(Values, G).to(device)

        policy_optimizer.zero_grad()
        policy_loss.sum().backward(retain_graph=True)
        policy_optimizer.step()

        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        v_loss_list = np.abs(errors.detach().cpu().numpy())
        p_loss_list = policy_loss.detach().cpu().numpy()

        Loss = v_loss_list + p_loss_list

        print(f"\n\nLevel: {maze_difficulty_level+1}/{max_difficulty_level+1} "
              f"\tRewards: {np.sum(Rewards):0.4f} ({np.mean(Rewards):0.4f}), States Visited: {len(visit_counts)}"
              f"\nPolicy Loss: {np.sum(p_loss_list):0.4f} ({np.mean(p_loss_list):0.4f})"
              f"\tValue Loss: {np.sum(v_loss_list):0.4f} ({np.mean(v_loss_list):0.4f})"
              f"\nNew states: {new_state_counter} / {h + 1} ({round(100*new_state_counter/(h + 1), 2)}%) "
              f"\tBack-step states: {back_step_counter} / {h + 1} ({round(100*back_step_counter/(h + 1), 2)}%) "
              f"\tWall-hits: {wall_step_counter} / {h + 1} ({round(100*wall_step_counter/(h + 1), 2)}%)\n")

        if exists(model_filepath):
            shutil.copy2(model_filepath, 'backups/')

        torch.save(nn.state_dict(), model_filepath)

        if len(returns) >= returns_history_length and np.mean(returns)/reward_success > 0.85:
            print('\n\n Increasing environment difficulty! \n\n')
            maze_difficulty_level = min(maze_difficulty_level+1, max_difficulty_level)
            env.close()
            variables_and_networks = load_env_and_model(maze_difficulty_level, agent_policy_alpha, agent_value_alpha)

            env = variables_and_networks[0]
            max_steps = variables_and_networks[1]
            nn = variables_and_networks[2]
            policy_optimizer = variables_and_networks[3]
            value_optimizer = variables_and_networks[4]

            env.reset()
            returns = deque([], maxlen=returns_history_length)
            k = 0

        continue

        cumulative_returns.append(sum(Rewards))
        cumulative_losses.append(sum(Loss))
        # Plot stuff
        plt.ioff()
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
        plt.savefig('REINFORCE_2.png')
        plt.clf()
        plt.close()

    cv2.destroyAllWindows()
    env.close()
