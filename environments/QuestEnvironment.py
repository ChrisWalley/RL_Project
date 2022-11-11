import gym
import cv2
import pyglet

from nle import nethack
from minihack import RewardManager
from gym.envs.classic_control import rendering

SEED = 42

# wrapper for rendering the env as an image
class RenderingWrapper(gym.Wrapper):

    def __init__(self, env, seed):
        super().__init__(env)
        self.env = env
        self.seed_value = seed
        self.viewer = rendering.SimpleImageViewer()
        self.viewer.width = 1280
        self.viewer.height = 520
        self.viewer.window = pyglet.window.Window(
            width=self.viewer.width, 
            height=self.viewer.height,
            display=self.viewer.display, 
            vsync=False, 
            resizable=True
        )
        self.action_history = []
        self.pixel_frames = []
        self.episode = 0

    def revert(self, actions=None):

        state = self.env.reset()

        if actions:
            for action in actions:
                _ = self.env.step(action)
        else:
            if self.action_history:
                self.action_history.pop()

            for action in self.action_history:
                state, _, _, _ = self.env.step(action)

        return state

    def clone(self):

        env_clone = QuestEnvironment().create(
            reward_lose = -10,
            reward_win = 10,
            penalty_step = -0.002,
            penalty_time = -0.001
        )

        env_clone.reset()

        # take all the actions the other environment took
        for action in self.action_history:
            _ = env_clone.step(action)

        return env_clone

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.action_history.append(action)
        self.pixels = obs['pixel']
        self.pixel_frames.append(self.pixels)
        #return obs['glyphs_crop'], reward, done, info
        return obs, reward, done, info

    def render(self, mode="human", **kwargs):
        if mode == 'human':
            self.viewer.imshow(self.pixels)
            return self.viewer.isopen
        else:
            return self.env.render()

    def reset(self, save_video=False, seed = None):

        self.episode += 1

        if save_video and len(self.pixel_frames) > 0:

            #save the video
            video_name = f'episode_{self.episode}.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(video_name, fourcc, 30, (self.pixel_frames[0].shape[1], self.pixel_frames[0].shape[0]))
            for frame in self.pixel_frames:
                video.write(frame[:,:,::-1])

            video.release()
        
        self.pixel_frames.clear()

        obs = self.env.reset()

        if seed:
            self.seed_value = seed
            
        self.env.seed(self.seed_value)
        self.pixels = obs['pixel']

        # TODO: make sure this is ok
        #return obs['glyphs_crop']
        return obs

    def close(self):
        if self.viewer is not None:
            self.viewer.window.close()
            self.viewer.close()
            self.viewer = None


class QuestEnvironment:

    def __init__(self):
        
        self.visit_counts = dict()
        self.coord_rewards = dict()

    def message_reward(self, env, obs, action, nobs):

        message = obs[5]
        msg = bytes(message)
        msg = str(msg).replace("'", "").replace('"', '').lstrip("b").rstrip("\\x00").rstrip("\x00")

        if msg == 'The door opens.':
            return 10

        if msg in [
            'You dont have anything to eat.',
            'Its solid stone.',
            'There is nothing here to pick up.',
            'You dont have anything to use or apply.'
        ]:
            return -0.00001

        return 0

    def _get_actions(self):

        return (
            nethack.CompassCardinalDirection.E,
            nethack.CompassCardinalDirection.N,
            nethack.CompassCardinalDirection.S,
            nethack.CompassCardinalDirection.W,
            nethack.MiscDirection.DOWN,
            nethack.Command.EAT,
            #nethack.MiscDirection.UP,
            #nethack.Command.OPEN,
            #nethack.Command.MOVE,
            nethack.Command.PICKUP,
            # nethack.Command.CAST,
            # nethack.Command.TELEPORT,
            # nethack.Command.WIELD, 
            #nethack.Command.SEARCH,
            # nethack.Command.KICK,
            #nethack.Command.LOOK, 
            # nethack.Command.JUMP, 
            # nethack.Command.SWAP,
            nethack.Command.ZAP,
            # nethack.Command.LOOT,
            # nethack.Command.PUTON,
            nethack.Command.APPLY,
            # nethack.Command.CAST,
            # nethack.Command.DIP,
            # nethack.Command.READ,
            # nethack.Command.INVOKE,
            # nethack.Command.RUSH,
            # nethack.Command.WEAR,
            # nethack.Command.ENHANCE,
            # nethack.Command.MOVEFAR,
            # nethack.Command.FIGHT
        )

    # create the environment
    # https://minihack.readthedocs.io/en/latest/envs/skills/quest.html
    def create(
            self,
            reward_lose = -10,
            reward_win = 10,
            penalty_step = -0.002,
            penalty_time = 0.002,
            max_episode_steps = 5000,
            seed=SEED
        ):

        import numpy as np

        self.visited_states_map = np.zeros((21,79)) # a map of counts for each state
        self.previous_coordinates = (0,0)

        # setup the reward manager
        # https://minihack.readthedocs.io/en/latest/getting-started/reward.html?highlight=RewardManager#reward-manager
        reward_manager = RewardManager()
        reward_manager.add_kill_event("minotaur", reward=10)

        # reward not bumping into things...
        #reward_manager.add_message_event('', 0.0001)

        reward_manager.add_custom_reward_fn(self.message_reward)


        # make the environment
        env = gym.make(
            "MiniHack-Quest-Hard-v0",
            actions = self._get_actions(),
            reward_manager = reward_manager,
            observation_keys = ("glyphs", "pixel", "glyphs_crop", "blstats", "pixel_crop", "chars_crop", "message", "screen_descriptions"),
            reward_lose = reward_lose,
            reward_win = reward_win,
            penalty_step = penalty_step,
            penalty_time = penalty_time,
            max_episode_steps = max_episode_steps,
            obs_crop_h=9,
            obs_crop_w=9,
        )

        env = RenderingWrapper(env, seed)

        #env = gym.wrappers.RecordVideo(env, "./video", episode_trigger = lambda x: x % 2 == 0)
        #env = gym.wrappers.Monitor(env, './video', video_callable=lambda x: x % 2 == 0, force=True)

        #print(f"Number of actions: {env.action_space.n}")

        return env