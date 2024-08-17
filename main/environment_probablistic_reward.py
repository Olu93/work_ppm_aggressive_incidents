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
import joblib

class TaskEnvProbablisticTimePenalty(TaskEnv):
    def __init__(self, time_out: int = 6, timeout_reward=-1, goal_reward=1, invalid_reward=-1, time_reward_multiplicator=0.01, frequencies_file=None, time_probabilities_file=None):
        super().__init__(time_out, timeout_reward, goal_reward, invalid_reward, time_reward_multiplicator, frequencies_file)
        self.time_probs = json.load(io.open(time_probabilities_file, 'r'))
        

    def step(self, action: int) -> Tuple[int, float, bool, object]:
        current_state = self.current_position

        valid = self._is_valid(current_state, action)
        self.episode_actions.append((action, "VALID" if valid else "INVALID"))
        next_state = self._get_next_state(self.current_position, action)

        days_elapsed = self.sample_days_past(current_state, action, next_state)

        step_sequence = (self.idx2inc[current_state],
                         self.idx2act[action], self.idx2inc[next_state])

        reward = self.reward_function(current_state, action, next_state)
        done = True if self._is_goal(next_state) else False

        self.timer += days_elapsed
        self.current_position = next_state
        return current_state, reward, done, {
            "step_sequence": step_sequence
        }


    def reward_function(
            self, state: int, action: int,
            next_state: int, **kwargs) -> Tuple[int, float, bool, object]:
        
        past_days = kwargs.get('past_days')

        valid = self._is_valid(state, action)

        incident_penalty = self.severity[self.idx2inc[next_state]]
        action_penalty = self.action_reward[self.idx2act[action]]

        if valid:
            # TODO: Need to define validity!!!
            pass

        if self._is_timeout():
            reward = self.timeout_reward
        elif self._is_goal(next_state):
            reward = incident_penalty + action_penalty + (self.time_reward_multiplicator * past_days)
        elif not valid:
            reward = self.invalid_reward
        else:
            reward = incident_penalty + action_penalty + (self.time_reward_multiplicator * past_days)

        return reward

    def sample_days_past(self, incident,  action, reaction):
        try:
            return stats.geom.rvs(self.time_probs[self.idx2inc[incident]][self.idx2act[action]][self.idx2inc[reaction]], size=1)[0]
        except KeyError as e:
            # print(e)
            return stats.geom.rvs(self.time_probs["Other"]["Other"]["Other"], size=1)[0]

    def transition_probability(self, state: int, action: int,  next_state: int):
        p = self.p_matrix[state, action, next_state]
        return p

    def _get_next_state(self, state, action):
        p_trans = self.p_matrix[state, action]

        return np.random.choice(len(p_trans), size=1, p=p_trans)[0]

# ===============================================================================

class TaskEnv2StepProbablisticTimePenalty(TaskEnv):
    def __init__(self, time_out: int = 6, timeout_reward=-1, goal_reward=1, invalid_reward=-1, time_reward_multiplicator=0.1, frequencies_file=None, time_probabilities_file=None, classification_pipeline=None):
        super().__init__(time_out, timeout_reward, goal_reward, invalid_reward, time_reward_multiplicator, frequencies_file)
        self.time_probs = json.load(io.open(time_probabilities_file, 'r'))
        self.next_state_classification_pipeline = joblib.load(classification_pipeline)
        

    def step(self, action: int) -> Tuple[int, float, bool, object]:
        current_state = self.current_position

        valid = self._is_valid(current_state, action)
        self.episode_actions.append((action, "VALID" if valid else "INVALID"))

        days_elapsed = self.sample_days_past(current_state, action)
        next_state = self._get_next_state(current_state, action, past_days=days_elapsed)


        step_sequence = (self.idx2inc[current_state],
                         self.idx2act[action], self.idx2inc[next_state])

        reward = self.reward_function(current_state, action, next_state, past_days=days_elapsed)
        done = True if self._is_goal(next_state) or self._is_timeout() else False
        
        self.timer += days_elapsed
        self.current_position = next_state

        return current_state, reward, done, {
            "step_sequence": step_sequence
        }

    def reward_function(
            self, state: int,  action: int,
            next_state: int, **kwargs) -> Tuple[int, float, bool, object]:
        
        past_days = kwargs.get('past_days')

        valid = self._is_valid(state, action)

        incident_penalty = self.severity[self.idx2inc[next_state]]
        action_penalty = self.action_reward[self.idx2act[action]]

        if valid:
            # TODO: Need to define validity!!!
            pass

        if self._is_timeout():
            reward = self.timeout_reward
        elif self._is_goal(next_state):
            reward = incident_penalty + action_penalty + (self.time_reward_multiplicator * past_days)
        elif not valid:
            reward = self.invalid_reward
        else:
            reward = incident_penalty + action_penalty + (self.time_reward_multiplicator * past_days)

        return reward


    def transition_probability(self,  state: int, action: int, next_state: int):
        p = self.p_matrix[state, action, next_state]
        return p

    def sample_days_past(self, incident, action):
        params = self.time_probs[self.idx2inc[incident]][self.idx2act[action]]
        shape = params.get('shape')
        loc = params.get('loc')
        scale = params.get('scale')
        return stats.weibull_min.rvs(shape, loc=loc, scale=scale, size=1)[0]

    def _get_next_state(self, state, action, **kwargs):
        past_days = kwargs.get('past_days')

        p_trans = self.next_state_classification_pipeline.predict_proba(pd.DataFrame([{
            "Aggression_short":self.idx2inc[state],
            "agent_action":self.idx2act[action],
            "Next_DaysToNext":past_days,
        }]))[0]

        chosen_next_state = np.random.choice(len(p_trans), size=1, p=p_trans)[0] 
        chosen_class = self.next_state_classification_pipeline['classifier'].classes_[chosen_next_state]
        return self.inc2idx[chosen_class]
