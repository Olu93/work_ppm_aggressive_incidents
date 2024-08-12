import ast
from typing import List, Tuple

import numpy as np
import gym
from gym.spaces import Box
from gym.spaces import Discrete
from PIL import Image as PImage
from gym.utils import seeding
import random
import pandas as pd


class TaskEnv(gym.Env):
    env_id = 'RandomMaze-v0'
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 3
    }
    reward_range = (-float('inf'), float('inf'))

    def __init__(self,
                 time_out: int = 6,
                 timeout_reward=-1,
                 goal_reward=1,
                 invalid_reward=-1,
                 time_reward_multiplicator=.01,
                 frequencies_file=None):
        """Contructor for the TaskEnvironment

        Args:
            size (int, optional): The size of the maze. Defaults to 15.
            time_out (int, optional): Time to explore the maze before the game is over. Defaults to 100.
        """
        super().__init__()

        self.motions = [
            'contact beeindigd/weggegaan',
            'client toegesproken/gesprek met client',
            'geen',
            'client afgeleid',
            'naar andere kamer/ruimte gestuurd',
            'met kracht tegen- of vastgehouden',
            'afzondering (deur op slot)',
        ]
        self.incidents = ['va', 'pp', 'po', 'sib']

        self.severity = {
            'va': 0.0,
            'po': -1.0,
            'sib': -3.0,
            'pp': -4.0,
            'Tau': 1.0
        }

        self.action_reward = {
            'contact beeindigd/weggegaan': -1.0,
            'client toegesproken/gesprek met client': 0,
            'geen': 0,
            'client afgeleid': -1.0,
            'naar andere kamer/ruimte gestuurd': -1.0,
            'met kracht tegen- of vastgehouden': -2.0,
            'afzondering (deur op slot)': -2.0,
        }
        # self.action_reward = {
        #     'contact beeindigd/weggegaan': -1.0,
        #     'client toegesproken/gesprek met client': -1.0,
        #     'geen': -1.0,
        #     'client afgeleid': -1.0,
        #     'naar andere kamer/ruimte gestuurd': -1.0,
        #     'met kracht tegen- of vastgehouden': -1.0,
        #     'afzondering (deur op slot)': -1.0,
        # }
        # self.action_reward = {
        #     'contact beeindigd/weggegaan': 0,
        #     'client toegesproken/gesprek met client': 0,
        #     'geen': 0,
        #     'client afgeleid': 0,
        #     'naar andere kamer/ruimte gestuurd': 0,
        #     'met kracht tegen- of vastgehouden': 0,
        #     'afzondering (deur op slot)': 0,
        # }

        frequencies: pd.DataFrame = pd.read_csv(frequencies_file, index_col=0)

        self.frequencies: pd.DataFrame = frequencies.map(ast.literal_eval)

        num_actions, num_incidents = self.frequencies.shape

        # NOTE: Dim are num of actions, incidents and then following incidents
        self.p_matrix = np.zeros(
            (num_incidents, num_actions, num_incidents + 1))
        self.act2idx = {act: idx for idx, act in enumerate(self.motions)}
        self.idx2act = dict(zip(self.act2idx.values(), self.act2idx.keys()))
        self.inc2idx = {inc: idx for idx, inc in enumerate(self.incidents)}
        self.inc2idx["Tau"] = len(self.inc2idx)
        self.idx2inc = dict(zip(self.act2idx.values(), self.inc2idx.keys()))

        for m in self.motions:
            for i in self.incidents:
                for key, val in self.frequencies.loc[m, i].items():
                    self.p_matrix[self.inc2idx[i], self.act2idx[m],
                                  self.inc2idx[key]] = val

        self.viewer = None

        self.time_out = time_out
        self.timer = 0
        self.timeout_reward = timeout_reward

        self.goal = self.inc2idx["Tau"]
        self.goal_reward = goal_reward

        self.invalid_reward = invalid_reward
        self.time_reward_multiplicator = time_reward_multiplicator
        self.seed()

        # Explore spaces
        self.observation_space = Discrete(len(self.incidents))
        self.action_space = Discrete(len(self.motions))
        self.current_position = self.observation_space.sample()
        self.episode_actions = []

    def step(self, action: int) -> Tuple[int, float, bool, object]:
        motion = self.motions[action]

        valid = self._is_valid(self.current_position, action)
        self.episode_actions.append((action, "VALID" if valid else "INVALID"))
        new_position = self._get_next_state(self.current_position, action)

        incident_penalty = self.severity[self.idx2inc[new_position]]
        action_penalty = self.action_reward[self.idx2act[action]]

        step_sequence = (self.idx2inc[self.current_position],
                         self.idx2act[action], self.idx2inc[new_position])
        if valid:
            # TODO: Need to define validity!!!
            pass

        if self._is_timeout():
            reward = self.timeout_reward
            done = True
        elif self._is_goal(new_position):
            reward = incident_penalty + action_penalty
            done = True
        elif not valid:
            reward = self.invalid_reward
            done = False
        else:
            reward = incident_penalty + action_penalty
            # reward = reward - (self.timer * self.time_reward_multiplicator)
            done = False
        self.timer += 1
        self.current_position = new_position
        return self.current_position, reward, done, {
            "step_sequence": step_sequence
        }

    def reward_function(
            self, action: int, state: int,
            next_state: int) -> Tuple[int, float, bool, object]:

        valid = self._is_valid(state, action)

        incident_penalty = self.severity[self.idx2inc[next_state]]
        action_penalty = self.action_reward[self.idx2act[action]]

        if valid:
            # TODO: Need to define validity!!!
            pass

        if self._is_timeout():
            reward = self.timeout_reward
        elif self._is_goal(next_state):
            reward = incident_penalty + action_penalty
        elif not valid:
            reward = self.invalid_reward
        else:
            reward = incident_penalty + action_penalty

        return reward

    def transition_probability(self, action: int, state: int, next_state: int):
        p = self.p_matrix[state, action, next_state]
        return p

    def reset(self) -> Tuple[int, int]:
        """Resets the environment. The agent will be transferred to a random location on the map. The goal stays the same and the timer is set to 0.

        Returns:
            Tuple[int, int]: The initial position of the agent.
        """
        self.timer = 0
        self.current_position = self.observation_space.sample()
        return self.current_position

    def _get_next_state(self, state, action):
        p_trans = self.p_matrix[state, action]

        return np.random.choice(len(p_trans), size=1, p=p_trans)[0]

    def _is_valid(self, incident, action) -> bool:
        """Checks if the position belongs to a wall or other disturbance

        Args:
            position (np.ndarray): The position whose validity to check

        Returns:
            bool: Validity is True for positions that are free and False for impassable positions
        """
        return True

    def _is_goal(self, incident: np.ndarray) -> bool:

        return self.goal == incident

    def _is_timeout(self) -> bool:
        """Checks whether the environment has reached its timeout.

        Returns:
            bool: True for timeout is exceeded and false if not.
        """
        return self.timer >= self.time_out

    def get_image(self) -> np.ndarray:
        """Helper for render function that returns an image of the current environment.

        Returns:
            np.ndarray: An array with the shape [height, width, rgb] image with values from 0 to 255
        """
        return self.maze.to_rgb()

    def seed(self, seed: int = None) -> List[int]:
        """Ensures reproductability

        Args:
            seed (int, optional): A seed number. Defaults to None.

        Returns:
            List[int]: The seed
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode: str = 'rgb_array') -> np.ndarray:
        """Renders the environment and returns either the img array or starts a live viewer.

        Args:
            mode (str, optional): Either "rgb_array" or "human". Defaults to 'rgb_array'.

        Returns:
            np.ndarray: The image
        """
        img = self.get_image()
        img = np.asarray(img).astype(np.uint8)
        img_height, img_width = img.shape[:2]
        ratio = self.observation_space.shape[0] / img_width
        img = PImage.fromarray(img).resize(
            [int(ratio * img_width),
             int(ratio * img_height)])
        img = np.asarray(img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control.rendering import SimpleImageViewer
            if self.viewer is None:
                self.viewer = SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# def register_environment(cls):
#     env_dict = gym.envs.registration.registry.env_specs.copy()
#     for env in env_dict:
#         if cls.env_id in env:
#             print("Remove {} from registry".format(env))
#             del gym.envs.registration.registry.env_specs[env]
#     gym.envs.register(id=cls.env_id, entry_point=cls, max_episode_steps=200)

# register_environment(TaskEnv)
