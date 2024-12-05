import gymnasium as gym
import math
import numpy as np

from collections import deque
from game import Game
from auxilary import Move, Rescuer
from typing import List, Tuple
from arcade import SpriteList, Sprite


class Environment(gym.Env):
    def __init__(self, screen_width, screen_height, screen_title):
        super(Environment, self).__init__()
        self.screen_width = screen_width

        # Create game environment
        self.game: Game = Game(screen_width, screen_height, screen_title)
        self.game.setup()

        # Define the numerical part of the observation space
        self.numerical_obs_space = gym.spaces.Box(
            low=0, high=1, shape=(4,), dtype=np.float32)
        # Define the image part of the observation space
        self.image_obs_space = gym.spaces.Box(
            low=0, high=1, shape=(
                300, 300), dtype=np.uint8)

        # Combine them into a dictionary-based observation space
        self.observation_space = gym.spaces.Dict(
            {'numerical': self.numerical_obs_space,
             'image': self.image_obs_space})

        # Define actions
        self.action_space = gym.spaces.MultiDiscrete([4])

        # Movement of the Rescuer
        self.action_mapping = {
            0: (-250, 0),
            1: (0, -250),
            2: (250, 0),
            3: (0, 250),
        }

        self.prev_movement_distance: float | None = None
        self.prev_avoidance_distance: float | None = None

        self.movement_reward_queue = deque()
        self.avoidance_reward_queue = deque()

    def reset(self, seed=None):
        # Return the initial observation
        self.game.reset()
        self.game.custom_draw()
        self.game.dispatch_events()
        self.game.flip()

        self.prev_movement_distance = None
        self.prev_avoidance_distance = None
        self.movement_reward_queue.clear()
        self.avoidance_reward_queue.clear()

        obs = self.get_obs()
        return obs, {}

    def close(self):
        self.game.close()

    def reward_function(self, obs: List) -> float:
        """
        Calculate the reward for an agent based on its movement towards a target 
        and its ability to avoid asteroids.

        Parameters:
        observation (list): A list representing the current observation

        Returns:
        float: A reward

        The function consists of two main components:
        1. Movement Reward:
        - The agent is rewarded for moving towards the target 
        and punished for moving away.

        - The reward is calculated using the Euclidean (L2) distance 
        between the agent's previous and current positions
        relative to the target.

        2. Asteroid Avoidance Reward:
        - The agent is punished for moving closer to asteroids and rewarded for moving away.
        - This reward is only active if an asteroid is within the vicinity of the agent.

        The reward is a weighted sum of both rewards.
        """

        ########################################################
        ############### Rescuer movement reward ################
        ########################################################

        obs = obs["numerical"]
        curr_dist = self._euclidean_distance(obs[0], obs[1], obs[2], obs[3])

        if self.prev_movement_distance is not None:
            progress = self.prev_movement_distance - curr_dist
        else:
            progress = 0
        self.prev_movement_distance = curr_dist

        num_movement_rewards = len(
            self.movement_reward_queue) if self.movement_reward_queue else 1

        # Maintain a maximum of 10 rewards in the queue
        if num_movement_rewards == 10:
            self.movement_reward_queue.popleft()
        self.movement_reward_queue.append(progress)

        # Calculate the average reward
        total = sum(self.movement_reward_queue)
        movement_reward = 0.5 * progress + 0.5 * (total / num_movement_rewards)

        ########################################################
        ############### Astroid avoidance reward ###############
        ########################################################

        rescuer: Rescuer = self.game.rescuer_list[0]
        astroids: SpriteList = self.game.asteroids_list
        distances = []

        for astroid in astroids:
            dist = self._get_distance_between_sprites(rescuer, astroid)
            if dist < 130:
                distances.append(dist)

        if not distances:
            # Clear list, since no astroids in vicinity
            self.avoidance_reward_queue.clear()
            progress = 0
        else:
            # Agent is rewarded if he moves away from astroid
            curr_dist = min(distances)
            if self.prev_avoidance_distance is not None:
                progress = curr_dist - self.prev_avoidance_distance
            else:
                progress = 0
            self.prev_avoidance_distance = curr_dist

        num_avoidance_rewards = len(
            self.avoidance_reward_queue) if self.avoidance_reward_queue else 1

        # Maintain a maximum of 10 rewards in the queue
        if num_avoidance_rewards == 10:
            self.avoidance_reward_queue.popleft()
        self.avoidance_reward_queue.append(progress)

        total = sum(self.avoidance_reward_queue)

        avoidance_reward = 0.5 * progress + \
            0.5 * (total / num_avoidance_rewards)

        # Scaling to match magnitude
        return 1000 * movement_reward + avoidance_reward

    def step(self, actions: List[int]) -> Tuple:
        """
        Perform a single step in the gym environment based on the provided actions.

        This function takes a list of actions and executes a step in the environment.
        It returns the current observation, the reward, whether the episode is done,
        whether the episode is truncated, and additional info.

        Parameters:
        actions (List[int]): A list of actions to be taken in the environment.

        Returns:
        Tuple: A tuple containing:
            - obs: Coordination information and image of vicinity
            - reward: The reward obtained from the actions.
            - done: A boolean indicating whether the episode has ended.
            - truncated: A boolean indicating whether the episode was truncated.
            - info: Additional information about the environment.
        """
        reward = 0
        self.rescued_alfred = False

        rescuer: Rescuer = self.game.rescuer_list[0]
        has_carried_resource = rescuer.carries_resource

        # Process all actions
        for action in actions:
            force: tuple = self.action_mapping[action]
            movement_action: Move = Move(
                'MOVEMENT', rescuer, force)
            self.game.action_list.append(movement_action)

        made_mistake = self.game.custom_update()
        self.game.custom_draw()
        self.game.dispatch_events()
        self.game.flip()

        if has_carried_resource != rescuer.carries_resource and has_carried_resource:
            self.rescued_alfred = True

        done, obs, reward = self.decision(made_mistake)
        info = {}

        return obs, reward, done, False, info

    def decision(self, made_mistake: bool):
        """
        Parameters:
        made_mistake bool: A boolean whether the agent has made a mistake.
        (Collided with an astroid or wandered off the screen)

        Returns:
        Tuple:
            - obs: The current observation after performing the actions.
            - reward: The reward obtained from the actions.
            - done: A boolean indicating whether the episode has ended.
        """

        if made_mistake:
            obs = {}
            reward = -10
            done = True
        else:
            done = self.rescued_alfred
            obs = self.get_obs() if not done else {}
            reward = self.reward_function(obs) if not done else 10

        return done, obs, reward

    def get_obs(self):
        """
        Returns: An observation of the environment in dict format.
        Dict:
            - numerical: Contains the information about the target and the agent's
            current position
            - image: Contains a grey scale image of the agent's vicinity. 
            It is agent centered and of size 300x300
        """
        rescuer: SpriteList = self.game.rescuer_list[0]
        obs = []

        if rescuer.carries_resource:
            mothership: Sprite = self.game.mother_ship_list[0]
            obs = [mothership.center_x,
                   mothership.center_y,
                   rescuer.center_x,
                   rescuer.center_y]
        else:
            alien: Sprite = self.game.alien_list[0]
            obs = [alien.center_x,
                   alien.center_y,
                   rescuer.center_x,
                   rescuer.center_y]

        image_data = self.game.get_image(
            rescuer.center_x - 150,
            rescuer.center_y - 150,
            300,
            300)

        image_data = image_data.convert("L")
        image_data = np.array(image_data)

        return {
            "numerical": np.array(obs) /
            self.screen_width,
            "image": image_data /
            255.0}

    def _euclidean_distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def _initial_rescuer_alien_dist(self):
        rescuer: Rescuer = self.game.rescuer_list[0]
        mothership: Sprite = self.game.alien_list[0]
        return self._euclidean_distance(
            rescuer.center_x,
            rescuer.center_y,
            mothership.center_x,
            mothership.center_y)

    def _get_distance_between_sprites(self, rescuer: Rescuer, astroid: Sprite):
        # Get the points on the edges of the hitboxes
        points1 = rescuer.get_adjusted_hit_box()
        points2 = astroid.get_adjusted_hit_box()

        min_distance = float('inf')

        # Calculate the shortest distance between any point on sprite1 and any
        # point on sprite2
        for point1 in points1:
            for point2 in points2:
                distance = math.dist(point1, point2)
                if distance < min_distance:
                    min_distance = distance

        return min_distance
