import sys
from collections import deque

import cv2
from agents.AutoEncoder import Encoder, Decoder
from agents.PolicyValueNetwork import PolicyValueNetwork
import torch
import numpy as np
from environments.QuestEnvironment import QuestEnvironment

sys.path.append('../')
# if there is a Cuda GPU, then we want to use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_filepath = 'backups/model_weights.pth'
AE_encoder_model_filepath = 'backups/E_model_weights.pth'
AE_decoder_model_filepath = 'backups/D_model_weights.pth'
OBS_SPACE = 'glyphs_crop'
AE_OBS_SPACE = 'pixel_crop'
FULL_OBS_SPACE = 'glyphs'
frame_stack_size = 4
max_steps_per_level = [2000, 5000, 10000, 15000]
mazes = ['MiniHack-MazeWalk-9x9-v0', 'MiniHack-MazeWalk-15x15-v0', 'MiniHack-MazeWalk-45x19-v0', 'MiniHack-Quest-Hard-v0']


def load_env_and_models(difficulty_level):

    environment = QuestEnvironment().create(
        env_type=mazes[difficulty_level],
        reward_lose=-1,
        reward_win=1,
        penalty_step=-0.1,
        penalty_time=0,
        max_episode_steps=max_steps_per_level[difficulty_level],
    )

    latent_space_dim = frame_stack_size * np.prod(environment.observation_space.spaces[OBS_SPACE].shape)

    # One-hot coordinate encodings, to represent the agent location
    one_hot_coords_length = np.prod(environment.observation_space.spaces[FULL_OBS_SPACE].shape)

    agent = PolicyValueNetwork(environment.action_space.n, latent_space_dim + 2 * one_hot_coords_length).to(device)

    # Calculate the input dimensions for the autoencoder
    AE_input_shape_dim = environment.observation_space.spaces[AE_OBS_SPACE].shape

    # Reshape if needed
    if len(AE_input_shape_dim) == 2:
        AE_input_shape_dim = (AE_input_shape_dim[0], AE_input_shape_dim[1], 1)

    AE_encoder = Encoder(batch_size=frame_stack_size, encoded_space_dim=latent_space_dim,
                         image_size=frame_stack_size * AE_input_shape_dim).to(device)

    AE_encoder.load_state_dict(torch.load(AE_encoder_model_filepath))

    return environment, max_steps_per_level[difficulty_level], agent, AE_encoder, AE_input_shape_dim


if __name__ == "__main__":

    num_episodes = 1000
    seed = 42
    maze_difficulty_level = 3
    env, max_steps, nn, encoder, AE_input_shape = load_env_and_models(maze_difficulty_level)

    empty_frame = np.zeros((AE_input_shape[2], AE_input_shape[0], AE_input_shape[1]))
    frame_stack = deque([empty_frame] * frame_stack_size, maxlen=frame_stack_size)

    nn.eval()
    encoder.eval()
    with torch.no_grad():
        for k in range(num_episodes):
            obs = env.reset()
            done = False

            one_hot_visits = np.zeros(obs[FULL_OBS_SPACE].shape)
            one_hot_visits[int(obs['blstats'][1]), int(obs['blstats'][0])] = 1

            while not done:

                one_hot_coords = np.zeros(obs[FULL_OBS_SPACE].shape)
                one_hot_coords[int(obs['blstats'][1]), int(obs['blstats'][0])] = 1

                state_pixels = np.array(obs[AE_OBS_SPACE] / 255., dtype=np.float32).reshape(
                    (AE_input_shape[2], AE_input_shape[0], AE_input_shape[1]))

                frame_stack.append(state_pixels)
                feature_tensor = torch.tensor(np.array(frame_stack)).float().to(device)

                encoded_state = encoder(feature_tensor)
                state = torch.nn.functional.normalize(encoded_state.unsqueeze(0))
                state = torch.cat((state,
                                   torch.tensor(one_hot_visits.flatten()).float().to(device).unsqueeze(0),
                                   torch.tensor(one_hot_coords.flatten()).float().to(device).unsqueeze(0)), 1)
                action, probs = nn.action(state)
                value = nn.value(state)

                obs, r, done, info = env.step(action)
                env.render()

    cv2.destroyAllWindows()
    env.close()
