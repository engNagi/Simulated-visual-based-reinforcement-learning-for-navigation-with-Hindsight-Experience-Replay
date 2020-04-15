import random
import re
import ai2thor.controller
import numpy as np
import pandas as pd
import ai2thor.controller

random.seed(123)
np.random.seed(123)


#    This axis has “right hand” facing with respect to the forward Z-Axis,
#    Y-axis pointing upward, z-axis pointing forward, x axis  pointing to the left
class Environment_topview(object):

    def __init__(self,
                 distance,
                 reward_typ,
                 action_n=6,
                 grid_size=0.15,
                 player_screen_width=300,
                 player_screen_height=300,
                 full_scrn=False,
                 depth_image=False,
                 agent_mode="tall",
                 scene="FloorPlan225"):

        self.scene = scene
        self.action_n = action_n
        self.distance = distance
        self.grid_size = grid_size
        self.full_scrn = full_scrn
        self.agent_mode = agent_mode
        self.reward_typ = reward_typ
        self.depth_image = depth_image
        self.player_screen_width = player_screen_width
        self.player_screen_height = player_screen_height

        self.ctrl = ai2thor.controller.Controller(scene=self.scene,
                                                  gridSize=self.grid_size,
                                                  renderDepthImage=self.depth_image,
                                                  agentMode=self.agent_mode)

    def reset(self, x_pos, y_pos, z_pos, angle):

        self.ctrl.reset(self.scene)

        self.ctrl.step(action='TeleportFull', x=x_pos, y=y_pos, z=z_pos, rotation=angle, horizon=0.0)

        agent_position, _, agent_pose = self.agent_properties()

        self.ctrl.step(action="ToggleMapView")

        return agent_position, agent_pose

    def step(self, action, goal, distance):

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

        elif action == 3:
            self.ctrl.step(action="MoveBack")
            reward, done, distance_, first_person_obs, collision, agent_position, agent_pose = self.post_action_state(
                goal,
                distance)

        elif action == 4:
            self.ctrl.step(action="MoveRight")
            reward, done, distance_, first_person_obs, collision, agent_position, agent_pose = self.post_action_state(
                goal,
                distance)

        elif action == 5:
            self.ctrl.step(action="MoveLeft")
            reward, done, distance_, first_person_obs, collision, agent_position, agent_pose = self.post_action_state(
                goal,
                distance)

        ### This action is done to avoid any action that would disorient the TOP-View ####
        else:
            self.ctrl.step(action="Pass")
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
        positions = self.get_reachable_position()
        random_positions = random.sample(list(positions), 2)
        return random_positions[0], random_positions[1]

    def post_action_state(self, goal, dist):
        if self.reward_typ == "shaped":
            reward, done, dist_, first_person_obs, collide, agent_position, agent_pose = self.shaped_reward(goal=goal,
                                                                                                            dist=dist)
        else:
            reward, done, dist_, first_person_obs, collide, agent_position, agent_pose = self.sparse_reward(goal=goal)

        return reward, done, dist_, first_person_obs, collide, agent_position, agent_pose

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
            print("Top-V collided with " + object_name)
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
            print("Top-V collided with " + object_name)
            reward = -1
            done = True
        else:
            reward = -1
            done = False

        return reward, done, dist_, first_person_obs, collide, agent_position, agent_pose
