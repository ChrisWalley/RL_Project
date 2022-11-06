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
from environments.QuestEnvironment import QuestEnvironment

# resolve path for notebook
sys.path.append('../')
# if there is a Cuda GPU, then we want to use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_filepath = 'model_weights.pth'
AE_encoder_model_filepath = 'E_model_weights.pth'
AE_decoder_model_filepath = 'D_model_weights.pth'
OBS_SPACE = 'glyphs_crop'
AE_OBS_SPACE = 'pixel_crop'
max_steps_per_level = [2000, 5000, 10000, 15000]
mazes = ['MiniHack-MazeWalk-9x9-v0', 'MiniHack-MazeWalk-15x15-v0', 'MiniHack-MazeWalk-45x19-v0', 'MiniHack-Quest-Hard-v0']

reward_lose = -0.5
reward_win = 50
reward_escape_maze = 10
reward_exploration = 0.01
step_penalty = -0.0001
wall_penalty = -0.01


def load_env_and_models(difficulty_level):

    environment = QuestEnvironment().create(
        env_type=mazes[difficulty_level],
        reward_lose=reward_lose,
        reward_win=reward_win,
        penalty_step=wall_penalty,
        penalty_time=step_penalty,
        max_episode_steps=max_steps_per_level[difficulty_level],
    )
    latent_space_dim = 3*environment.observation_space.spaces[OBS_SPACE].shape[0] *\
                       environment.observation_space.spaces[OBS_SPACE].shape[1]

    agent = PolicyValueNetwork(environment.action_space.n, 0, latent_space_dim).to(device)

    agent.load_state_dict(torch.load(model_filepath))

    AE_input_shape_dim = environment.observation_space.spaces[AE_OBS_SPACE].shape

    if len(AE_input_shape_dim) == 2:
        AE_input_shape_dim = (AE_input_shape_dim[0], AE_input_shape_dim[1], 1)

    AE_encoder = Encoder(encoded_space_dim=latent_space_dim, fc2_input_dim=AE_input_shape_dim).to(device)
    AE_decoder = Decoder(encoded_space_dim=latent_space_dim, fc2_input_dim=AE_input_shape_dim).to(device)

    AE_encoder.load_state_dict(torch.load(AE_encoder_model_filepath))

    AE_decoder.load_state_dict(torch.load(AE_decoder_model_filepath))

    return environment, max_steps_per_level[difficulty_level], agent, AE_encoder, AE_decoder, AE_input_shape_dim


if __name__ == "__main__":

    num_episodes = 1000
    agent_alpha = 10e-5
    AE_alpha = 10e-5
    gamma = 0.9995
    seed = 42

    maze_difficulty_level = 3
    max_difficulty_level = 3

    env, max_steps, nn, encoder, decoder, AE_input_shape = load_env_and_models(maze_difficulty_level)
    nn.eval()
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for k in range(num_episodes):
            obs = env.reset()
            done = False
            escaped_maze = False
            while not done:
                state_pixels = np.array(obs[AE_OBS_SPACE] / 255., dtype=np.float32).reshape(
                   (AE_input_shape[2], AE_input_shape[0], AE_input_shape[1]))

                encoded_state = encoder(state_pixels)
                decoded_state = decoder(encoded_state.to(device))

                decoded_frame = decoded_state.detach().clone().cpu().numpy().reshape(AE_input_shape)
                combined = np.concatenate([decoded_frame, obs[AE_OBS_SPACE]/255.])

                cv2.imshow('Frame', combined)
                cv2.waitKey(1)

                coords = (int(obs['blstats'][0]), int(obs['blstats'][1]))
                escaped_maze = coords[0] > 29 and 8 < coords[1] < 14

                action, probs = nn.action(encoded_state)
                value = nn.value(encoded_state)

                obs, r, done, info = env.step(action)
                env.render()




    cv2.destroyAllWindows()
    env.close()
