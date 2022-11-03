import gym
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
        #return obs['glyphs_crop'], reward, done, info
        return obs, reward, done, info

    def render(self, mode="human", **kwargs):
        if mode == 'human':
            self.viewer.imshow(self.pixels)
            return self.viewer.isopen
        else:
            return self.env.render()

    def reset(self):
        obs = self.env.reset()
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

        if msg == b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00':
            return 0.1

        return 0

    def move_right_reward(self, env, obs, action, nobs):
        
        blstats = obs[4]
        coords = (blstats[0], blstats[1])

        if self.previous_coordinates[1] < coords[1]:
            self.previous_coordinates = coords
            return 0.00001
        else:
            return -0.00001


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
            #nethack.Command.PICKUP,
            # nethack.Command.CAST,
            # nethack.Command.TELEPORT,
            # nethack.Command.WIELD, 
            #nethack.Command.SEARCH,
            # nethack.Command.KICK,
            #nethack.Command.LOOK, 
            # nethack.Command.JUMP, 
            # nethack.Command.SWAP,
            # nethack.Command.ZAP,
            # nethack.Command.LOOT,
            # nethack.Command.PUTON,
            # nethack.Command.APPLY,
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
        #reward_manager.add_kill_event("goblin", reward=1)
        #reward_manager.add_kill_event("jackal", reward=1)
        #reward_manager.add_kill_event("giant rat", reward=1)
        
        #reward_manager.add_custom_reward_fn(self.move_right_reward)

        # reward not bumping into things...
        reward_manager.add_message_event('', 0.01)

        reward_manager.add_custom_reward_fn(self.message_reward)

        #reward_manager.add_message_event(
        #    [
        #        "What a strange direction!  Never mind.",
        #         "You don't have anything to eat.",
        #         "You can't go down here.",
        #         "It's solid stone.",
        #         "You faint from lack of food.  You regain consciousness."
        #    ],
        #    -0.001
        #)


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
            max_episode_steps = 100000000000000000,
            obs_crop_h=9,
            obs_crop_w=9,
        )

        env = RenderingWrapper(env, seed)

        #print(f"Number of actions: {env.action_space.n}")

        return env