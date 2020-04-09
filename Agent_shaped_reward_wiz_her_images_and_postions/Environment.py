import random
import re
import ai2thor.controller
import numpy as np
import pandas as pd
import ai2thor.controller

random.seed(123)
np.random.seed(123)


class Environment(object):
    def __init__(self,
                 distance,
                 reward_typ,
                 action_n=6,
                 grid_size=0.15,
                 visibility_distance=1.5,
                 player_screen_width=300,
                 player_screen_height=300,
                 full_scrn=False,
                 depth_image=False,
                 random_init_position=False,
                 random_init_pos_orient=False,
                 random_goals=False,
                 scene="FloorPlan225",
                 agent_mode="tall"):

        self.scene = scene
        self.action_n = action_n
        self.distance = distance
        self.grid_size = grid_size
        self.full_scrn = full_scrn
        self.reward_typ = reward_typ
        self.agent_mode = agent_mode
        self.depth_image = depth_image
        self.random_goal = random_goals
        self.visibility_distance = visibility_distance
        self.player_screen_width = player_screen_width
        self.player_screen_height = player_screen_height
        self.random_init_position = random_init_position
        self.random_init_pos_orient = random_init_pos_orient
        self.orientations = [0.0, 90.0, 180.0, 270.0, 360.0]

        self.ctrl = ai2thor.controller.Controller(scene=self.scene,
                                                  gridSize=self.grid_size,
                                                  renderDepthImage=self.depth_image,
                                                  visibilityDistance=self.visibility_distance,
                                                  agentMode=self.agent_mode)

    def reset(self):
        new_random_goal = 0

        self.ctrl.reset(self.scene)

        agent_init_position, random_goal = self.random_positions()

        if self.random_init_pos_orient:
            # Random init Agent positions and orientation
            self.ctrl.step(action="TeleportFull",
                           x=agent_init_position[0],
                           y=agent_init_position[1],
                           z=agent_init_position[2],
                           rotation=random.choice(self.orientations),
                           horizon=0.0)

        elif self.random_init_position:
            # Random init Agent positions only
            self.ctrl.step(action="Teleport",
                           x=agent_init_position[0],
                           y=agent_init_position[1],
                           z=agent_init_position[2])
        else:
            pass

        if self.random_goal:
            new_random_goal = random_goal

        agent_position, agent_rotation, agent_pose = self.agent_properties()

        try:
            np.array_equal(np.array(list(self.ctrl.last_event.metadata["agent"]["position"].values())), agent_position)
        except:
            print("agent init position does not equal to agent position attribute")

        pre_action_idx = 0
        # if the agent init_position equals the goal position respawn the agent in different position
        new_goal = self.agent_goal_pos_not_equal(agent_position, new_random_goal)

        agent_pos_dis = np.linalg.norm(new_goal - agent_position)

        first_person_obs = self.ctrl.last_event.frame

        return first_person_obs, agent_position, new_goal, agent_pos_dis, agent_pose, pre_action_idx

    def step(self, action, goal, distance):
        first_person_obs, agent_position, distance_, done, reward, collision, agent_pose = 0, 0, 0, 0, 0, 0, 0
        if action == 0:
            self.ctrl.step(action='RotateRight')
            reward, done, distance_, first_person_obs, collision, agent_position, agent_pose = self.post_action_state(
                goal,
                distance)

        elif action == 1:
            self.ctrl.step(action='RotateLeft')
            reward, done, distance_, first_person_obs, collision, agent_position, agent_pose = self.post_action_state(
                goal,
                distance)

        elif action == 2:
            self.ctrl.step(action="MoveAhead")
            reward, done, distance_, first_person_obs, collision, agent_position, agent_pose = self.post_action_state(
                goal,
                distance)

        return first_person_obs, agent_position, distance_, done, reward, collision, agent_pose

    def agent_properties(self):
        agent_position = np.array(list(self.ctrl.last_event.metadata["agent"]["position"].values()))
        agent_rotation = np.array(list(self.ctrl.last_event.metadata["agent"]["rotation"].values()))
        agent_pose = np.concatenate((agent_position, agent_rotation), axis=0)

        return agent_position, agent_rotation, agent_pose

    def get_reachable_position(self):
        self.ctrl.step(action='GetReachablePositions')
        return pd.DataFrame(self.ctrl.last_event.metadata["reachablePositions"]).values

    def random_positions(self):
        while True:
            positions = self.get_reachable_position()
            random_positions = random.sample(list(positions), 2)
            agent_pos = random_positions[0]
            goal_pos = random_positions[1]
            distance = np.linalg.norm(goal_pos - agent_pos)
            if distance > 1.5*self.distance:
                break
            else:
                print("Agent to Goal distance less than", 1.5*self.distance)
        return agent_pos, goal_pos

    def post_action_state(self, goal, dist):
        if self.reward_typ == "shaped":
            reward, done, dist_, first_person_obs, collide, agent_position, agent_pose = self.shaped_reward(goal=goal,
                                                                                                            dist=dist)
        else:
            reward, done, dist_, first_person_obs, collide, agent_position, agent_pose = self.sparse_reward(goal=goal)

        return reward, done, dist_, first_person_obs, collide, agent_position, agent_pose

    def agent_goal_pos_not_equal(self, agent_pos, goal_pos):
        new_random_goal_position = goal_pos
        distance = np.linalg.norm(goal_pos - agent_pos)
        if distance <= 1.5*self.distance:
            print("agent position and goal position < or =", 1.5*self.distance)
            _, new_random_goal_position = self.random_positions()
            print("agent new position:", agent_pos[0], ",", agent_pos[2], "and goal_position",
                  new_random_goal_position[0], ",", new_random_goal_position[2])
        return new_random_goal_position

    def shaped_reward(self, goal, dist):
        agent_position, agent_rotation, agent_pose = self.agent_properties()
        first_person_obs = self.ctrl.last_event.frame
        dist_ = np.linalg.norm(goal - agent_position)
        collide = not self.ctrl.last_event.metadata["lastActionSuccess"]
        if dist_ < self.distance:
            reward = 0
            done = True
        elif collide:
            error_message = self.ctrl.last_event.metadata["errorMessage"]
            collided_object = re.findall(r"\w*", error_message)
            object_name_ = collided_object[0].split("_")
            object_name = object_name_[0]
            print("Agent collided with " + object_name)
            reward = -1
            done = True
        elif dist_ < dist:
            reward = -0.1
            done = False
        elif dist_ == dist:
            reward = -0.5
            done = False
        else:
            reward = -1 + (dist_ - dist)
            done = False

        return reward, done, dist_, first_person_obs, collide, agent_position, agent_pose

    def sparse_reward(self, goal):
        agent_position, agent_rotation, agent_pose = self.agent_properties()
        first_person_obs = self.ctrl.last_event.frame
        dist_ = np.linalg.norm(goal - agent_position)
        collide = not self.ctrl.last_event.metadata["lastActionSuccess"]
        if dist_ <= self.distance:
            reward = 0
            done = True
        elif collide:
            error_message = self.ctrl.last_event.metadata["errorMessage"]
            collided_object = re.findall(r"\w*", error_message)
            object_name_ = collided_object[0].split("_")
            object_name = object_name_[0]
            print("Agent collided with " + object_name)
            reward = -1
            done = True
        else:
            reward = -1
            done = False

        return reward, done, dist_, first_person_obs, collide, agent_position, agent_pose
