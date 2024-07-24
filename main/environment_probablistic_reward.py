import ast
from typing import List, Tuple
from scipy import stats
import numpy as np
import gym
from gym.spaces import Box
from gym.spaces import Discrete
from PIL import Image as PImage
from gym.utils import seeding
import random
import pandas as pd
import json
from environment import TaskEnv
import io

class TaskEnvProbablisticTimePenalty(TaskEnv):
    def __init__(self, time_out: int = 6, timeout_reward=-1, goal_reward=1, invalid_reward=-1, time_reward_multiplicator=0.01, frequencies_file=None, time_probabilities_file=None):
        super().__init__(time_out, timeout_reward, goal_reward, invalid_reward, time_reward_multiplicator, frequencies_file)
        self.time_probs = json.load(io.open(time_probabilities_file, 'r'))
        

    def step(self, action: int) -> Tuple[int, float, bool, object]:
        motion = self.motions[action]

        valid = self._is_valid(self.current_position, action)
        self.episode_actions.append((action, "VALID" if valid else "INVALID"))
        new_position = self._get_next_state(self.current_position, action)

        sampled_days_past = self.sample_days_past(self.current_position, action, new_position)

        step_sequence = (self.idx2inc[self.current_position],
                         self.idx2act[action], self.idx2inc[new_position])

        incident_penalty = self.severity[self.idx2inc[new_position]]
        action_penalty = self.action_reward[self.idx2act[action]]

        if valid:
            # TODO: Need to define validity!!!
            pass

        if self._is_timeout():
            reward = self.timeout_reward
            done = True
        elif self._is_goal(new_position):
            reward = incident_penalty + action_penalty + sampled_days_past
            done = True
        elif not valid:
            reward = self.invalid_reward
            done = False
        else:
            reward = incident_penalty + action_penalty + sampled_days_past
            done = False

        self.timer += sampled_days_past
        self.current_position = new_position
        return self.current_position, reward, done, {
            "step_sequence": step_sequence
        }

    def sample_days_past(self, incident,  action, reaction):
        try:
            return stats.geom.rvs(self.time_probs[self.idx2inc[incident]][self.idx2act[action]][self.idx2inc[reaction]], size=1)[0]
        except KeyError as e:
            # print(e)
            return stats.geom.rvs(self.time_probs["Other"]["Other"]["Other"], size=1)[0]

    def transition_probability(self, action: int, state: int, next_state: int):
        p = self.p_matrix[state, action, next_state]
        return p

    def _get_next_state(self, state, action):
        p_trans = self.p_matrix[state, action]

        return np.random.choice(len(p_trans), size=1, p=p_trans)[0]




   

# def register_environment(cls):
#     env_dict = gym.envs.registration.registry.env_specs.copy()
#     for env in env_dict:
#         if cls.env_id in env:
#             print("Remove {} from registry".format(env))
#             del gym.envs.registration.registry.env_specs[env]
#     gym.envs.register(id=cls.env_id, entry_point=cls, max_episode_steps=200)

# register_environment(TaskEnv)
