from src.envs.action_wrappers import REDUCED_ACTION_MAP, ReducedActions
from src.envs.utils import find_mario, get_level

import numpy as np


class UniformActions:
    def __init__(self) -> None:
        self.actions_len = len(REDUCED_ACTION_MAP)

    def choose_action(self, observation):
        return np.random.randint(self.actions_len)


class HeuristicActions:
    def __init__(self):
        self.x = 0
        self.episode_counter = 0
        self.custom_actions = [
            ReducedActions.NOOP,
            ReducedActions.RIGHT,
            ReducedActions.LEFT,
            ReducedActions.UP,
            ReducedActions.JUMP,
        ]

    def update_x(self):
        self.episode_counter += 1
        if self.episode_counter % 2 == 0 and self.x < 5:
            self.x += 1

    def choose_action(self, observation):
        mario_y = find_mario(observation)[0]
        level = get_level(mario_y)

        rand_val = np.random.rand()
        if level == 0:  # Ground level
            if rand_val < 3 / 7:
                return ReducedActions.RIGHT  # Move right
            elif 3 / 7 <= rand_val < 5 / 7:
                return ReducedActions.UP  # Climb ladder (Move up)
            else:
                possible_actions = [
                    action
                    for action in self.custom_actions
                    if action not in [ReducedActions.RIGHT, ReducedActions.UP]
                ]
                return np.random.choice(possible_actions)
        elif level == 1:  # First level
            if rand_val < 3 / 7:
                return ReducedActions.LEFT  # Move left
            elif 3 / 7 <= rand_val < 5 / 7:
                return ReducedActions.UP  # Climb ladder (Move up)
            else:
                possible_actions = [
                    action
                    for action in self.custom_actions
                    if action not in [ReducedActions.LEFT, ReducedActions.UP]
                ]
                return np.random.choice(possible_actions)
        elif level == 2:  # Second level
            if rand_val < 3 / 7:
                return ReducedActions.RIGHT  # Move right
            elif 3 / 7 <= rand_val < 5 / 7:
                return ReducedActions.UP  # Climb ladder (Move up)
            else:
                possible_actions = [
                    action
                    for action in self.custom_actions
                    if action not in [ReducedActions.RIGHT, ReducedActions.UP]
                ]
                return np.random.choice(possible_actions)
        elif level == 3:  # Third level
            if rand_val < 3 / 7:
                return ReducedActions.LEFT  # Move left
            elif 3 / 7 <= rand_val < 5 / 7:
                return ReducedActions.UP  # Climb ladder (Move up)
            else:
                possible_actions = [
                    action
                    for action in self.custom_actions
                    if action not in [ReducedActions.LEFT, ReducedActions.UP]
                ]
                return np.random.choice(possible_actions)
        elif level == 4:  # Fourth level
            if rand_val < 3 / 7:
                return ReducedActions.RIGHT  # Move right
            elif 3 / 7 <= rand_val < 5 / 7:
                return ReducedActions.UP  # Climb ladder (Move up)
            else:
                possible_actions = [
                    action
                    for action in self.custom_actions
                    if action not in [ReducedActions.RIGHT, ReducedActions.UP]
                ]
                return np.random.choice(possible_actions)
