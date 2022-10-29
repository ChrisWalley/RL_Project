import gym
import pyglet

from nle import nethack
from minihack import RewardManager
from gym.envs.classic_control import rendering

# wrapper for rendering the env as an image
class RenderingWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.env = env
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

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.pixels = obs['pixel']
        return obs['glyphs_crop'], reward, done, info

    def render(self, mode="human", **kwargs):
        if mode == 'human':
            self.viewer.imshow(self.pixels)
            return self.viewer.isopen
        else:
            return self.env.render()

    def reset(self):
        obs = self.env.reset()
        self.pixels = obs['pixel']

        # TODO: make sure this is ok
        return obs['glyphs_crop']

    def close(self):
        if self.viewer is not None:
            self.viewer.window.close()
            self.viewer.close()
            self.viewer = None

class QuestEnvironment:

    def _get_actions(self):

        return tuple(nethack.CompassDirection) + (
            nethack.Command.SEARCH,
            nethack.Command.KICK,
            nethack.Command.OPEN,
            nethack.Command.LOOK, 
            nethack.Command.JUMP, 
            nethack.Command.PICKUP,
            nethack.Command.WIELD, 
            nethack.Command.SWAP,
            nethack.Command.EAT,
            nethack.Command.ZAP,
            nethack.Command.LOOT,
            nethack.Command.PUTON,
            nethack.Command.APPLY,
            nethack.Command.CAST,
            nethack.Command.DIP,
            nethack.Command.READ,
            nethack.Command.INVOKE,
            nethack.Command.RUSH,
            nethack.Command.WEAR,
            nethack.Command.ENHANCE
        )

    # create the environment
    # https://minihack.readthedocs.io/en/latest/envs/skills/quest.html
    def create(self):

        # setup the reward manager
        # https://minihack.readthedocs.io/en/latest/getting-started/reward.html?highlight=RewardManager#reward-manager
        reward_manager = RewardManager()
        reward_manager.add_kill_event("minotaur", reward=10)
        reward_manager.add_kill_event("goblin", reward=1)
        reward_manager.add_kill_event("jackal", reward=1)
        reward_manager.add_kill_event("giant rat", reward=1)

        # make the environment
        env = gym.make(
            "MiniHack-Quest-Hard-v0",
            actions = self._get_actions(),
            reward_manager = reward_manager,
            observation_keys = ("glyphs", "pixel", "glyphs_crop"),
            reward_lose = -10,
            reward_win = 10,
            penalty_step = -0.002,
            penalty_time = 0.002,
        )

        env = RenderingWrapper(env)

        return env