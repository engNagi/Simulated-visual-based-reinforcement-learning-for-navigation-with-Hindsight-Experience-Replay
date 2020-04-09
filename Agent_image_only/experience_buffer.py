import numpy as np
import random
import copy as cp


class experience_buffer(object):
    def __init__(self, distance, her_samples, reward_typ, mem_size=5000):
        self.memory = []
        self.mem_size = mem_size
        self.distance = distance
        self.her_samples = her_samples
        self.reward_typ = reward_typ

    def clear(self):
        self.memory = []

    def add(self, experience):
        if len(self.memory) + len(experience) >= self.mem_size:
            self.memory[0:(len(experience) + len(self.memory)) - self.mem_size] = []
        self.memory.extend(experience)

    def her(self):
        if self.reward_typ == "shaped":
            her_buffer = self.her_shaped()
        else:
            her_buffer = self.her_sparse()

        return her_buffer

    def her_shaped(self):
        her_buffer = cp.deepcopy([self.memory])
        for k in range(self.her_samples):
            future_samples = np.random.randint(0, len(self.memory))
            her_traj = cp.deepcopy(self.memory[0:future_samples + 1])
            #goal_state = her_traj[-1][4]  ##pre_action_idx, features, pos_state, curr_action_idx, reward, features_, pos_state, done, goal
            #goal_pos = goal_state[-3:]
            goal_pos = her_traj[-1][6]
            for trans in range(len(her_traj)):
                pre_action = her_traj[trans][0]
                state = her_traj[trans][1]
                pre_state_pos = her_traj[trans][2]
                action = her_traj[trans][3]
                next_state = her_traj[trans][5]
                next_state_pos = her_traj[trans][6]
                distance = np.linalg.norm(goal_pos - pre_state_pos)
                distance_ = np.linalg.norm(goal_pos - next_state_pos)
                if distance_ < self.distance:
                    reward = 0
                    done = True
                elif distance_ < distance:
                    reward = -0.1
                    done = False
                elif distance_ == distance:
                    reward = -0.2
                    done = False
                else:
                    reward = -1 + (distance_ - distance)
                    done = False
                her_traj[trans][0] = pre_action
                her_traj[trans][1] = state
                her_traj[trans][2] = pre_state_pos
                her_traj[trans][3] = action
                her_traj[trans][4] = reward
                her_traj[trans][5] = next_state
                her_traj[trans][6] = next_state_pos
                her_traj[trans][7] = done
                her_traj[trans][8] = goal_pos

            her_buffer.append(her_traj)

        return her_buffer

    def her_sparse(self):
        her_buffer = cp.deepcopy([self.memory])
        for k in range(self.her_samples):
            future_samples = np.random.randint(0, len(self.memory))
            her_traj = cp.deepcopy(self.memory[0:future_samples + 1])
            # goal_state = her_traj[-1][4]
            # goal_pos = goal_state[-3:]
            goal_pos = her_traj[-1][6]
            for trans in range(len(her_traj)):
                pre_action = her_traj[trans][0]
                state = her_traj[trans][1]
                pre_state_pos = her_traj[trans][2]
                action = her_traj[trans][3]
                next_state = her_traj[trans][5]
                next_state_pos = her_traj[trans][6]
                distance_ = np.linalg.norm(goal_pos - next_state_pos)
                if distance_ < self.distance:
                    reward = 0
                    done = True
                else:
                    reward = -1
                    done = False
                her_traj[trans][0] = pre_action
                her_traj[trans][1] = state
                her_traj[trans][2] = pre_state_pos
                her_traj[trans][3] = action
                her_traj[trans][4] = reward
                her_traj[trans][5] = next_state
                her_traj[trans][6] = next_state_pos
                her_traj[trans][7] = done
                her_traj[trans][8] = goal_pos
            her_buffer.append(her_traj)

        return her_buffer
