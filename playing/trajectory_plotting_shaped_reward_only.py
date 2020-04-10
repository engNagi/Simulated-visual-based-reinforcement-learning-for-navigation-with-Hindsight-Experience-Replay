import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib
from matplotlib.lines import Line2D

from playing.Environment import Environment
from playing.Environment_top_view import Environment_topview
from playing.autoencoder import load_autoencoder
from playing.DRQN_HER_Shaped_reward_only import DRQN
from playing.helper import train_valid_env_sync, trajectory_plots

matplotlib.use('TkAgg')

random.seed(123)
np.random.seed(123)

dir = "/home/nagi/Desktop/Master_project_final/final_training_wights_2/DRQN_3_shaped_reward_only_sequence/DRQN.ckpt"

##### environment_Variables
grid_size = 0.18  # size of the agent step
top_view = True  # displaying top-view
distance_threshold = grid_size * 2  # distance threshold to the goal
action_n = 3  # number of allowed action
random_init_position = False  # Random initial positions only -- no change in the agent orientation
random_init_pose = True  # Random initial positions with random agent orientation
reward = "shaped"  # reward type "shaped","sparse"

#########################   hyper-parameter
fcl_dims = 512
nodes_num = 256
## size of the input to the LSTM
input_size = 521

plotted_data = pd.DataFrame(columns=["Episodes", "Successful trajectories", "Failed trajectories", "Ratio", "Steps"])

legend_elements = [Line2D([0], [0], marker='o', color='white', label='Navigable Positions',
                          markerfacecolor='grey', markersize=10),
                   Line2D([0], [0], marker="X", color='white', label='Goal Position',
                          markerfacecolor='green', markersize=10),
                   Line2D([0], [0], marker="o", color='white', label='Initial Agent Position',
                          markerfacecolor='blue', markersize=10),
                   Line2D([0], [0], marker=">", color='white', label='Agent',
                          markerfacecolor='red', markersize=10)
                   ]

env = Environment(random_init_position=random_init_position, random_init_pos_orient=random_init_pose, reward_typ=reward,
                  distance=distance_threshold, random_goals=True, grid_size=grid_size, agent_mode="bot")

if top_view:
    envT = Environment_topview(grid_size=grid_size, agent_mode="bot", distance=distance_threshold, reward_typ=reward)
reachable_positions = env.get_reachable_position()

plt.ion()
successes = 0
failures = 0
total_steps = 0
alpha = 1

#   Autoenconder
print("Autoencoder")
ae_sess, ae = load_autoencoder()

print("DQN_HER_Model")
drqn_graph = tf.Graph()
cell = tf.nn.rnn_cell.LSTMCell(num_units=nodes_num, state_is_tuple=True)
cellT = tf.nn.rnn_cell.LSTMCell(num_units=nodes_num, state_is_tuple=True)

model = DRQN(action_n=action_n, cell=cell, fcl_dims=fcl_dims, scope="model",
             save_path=dir, nodes_num=nodes_num, input_size=input_size)
target_model = DRQN(action_n=action_n, cell=cellT, fcl_dims=fcl_dims, scope="target_model",
                    save_path=dir, nodes_num=nodes_num, input_size=input_size)

print("##### Env with grid_size equals", grid_size, "and", reward, "reward ######")

with tf.Session() as sess:
    model.set_session(sess)
    target_model.set_session(sess)
    sess.run(tf.global_variables_initializer())
    model.load()
    for i in range(3):
        step_num = 0

        #   rnn_init_state
        rnn_state = (np.zeros([1, nodes_num]), np.zeros([1, nodes_num]))
        # reset environment
        obs_state, pos_state, goal, distance, pose, pre_action_idx = env.reset()
        if top_view:
            # additional env top view for validation
            agent_pos_top, pose_top = envT.reset(x_pos=pose[0],
                                                 y_pos=pose[1],
                                                 z_pos=pose[2],
                                                 angle=pose[4])

        features = ae_sess.run(ae.feature_vector, feed_dict={ae.image: obs_state[None, :, :, :]})
        features = np.squeeze(features, axis=0)
        obs_pos_state = np.concatenate((features, pos_state), axis=0)

        plt.close()
        plt.figure()
        plt.ion()
        for pos in reachable_positions:
            plt.scatter(pos[0], pos[2], s=20, c="grey", marker="o", alpha=1)
        x_start, x_end = plt.xlim()
        y_start, y_end = plt.ylim()
        plt.yticks(np.arange((y_start - 0.18), y_end, 0.18))
        x = np.arange(x_start, x_end, 0.18)
        plt.xticks(np.arange(x_start, x_end, 0.18), rotation=90)
        plt.xlabel("Z-Coordinates")
        plt.ylabel("X-Coordinates")
        plt.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1), prop={'size': 6})
        plt.grid()
        plt.tight_layout()
        # plt.draw()
        plt.scatter(pos_state[0], pos_state[2], c="blue", marker="o")
        plt.scatter(goal[0], goal[2], c="green", marker="X")
        # additional env top view for validation

        plt.pause(0.9)
        if pose[4] == 0:
            plt.scatter(pos_state[0], pos_state[2], c="white", marker="s", alpha=1)
            plt.scatter(pos_state[0], pos_state[2], c="red", marker="v")
        elif pose[4] == 90:
            plt.scatter(pos_state[0], pos_state[2], c="white", marker="s", alpha=1)
            plt.scatter(pos_state[0], pos_state[2], c="red", marker=">")
        elif pose[4] == 180:
            plt.scatter(pos_state[0], pos_state[2], c="white", marker="s", alpha=1)
            plt.scatter(pos_state[0], pos_state[2], c="red", marker="^")
        elif pose[4] == 270:
            plt.scatter(pos_state[0], pos_state[2], c="white", marker="s", alpha=1)
            plt.scatter(pos_state[0], pos_state[2], c="red", marker="<")

        plt.pause(0.01)

        done = False
        while not done:
            # clear_output(wait=True)
            curr_action_idx, rnn_state_ = model.sample_action(goal=goal,
                                                              batch_size=1,
                                                              trace_length=1,
                                                              epsilon=0,
                                                              rnn_state=rnn_state,
                                                              pos_obs_state=obs_pos_state,
                                                              pre_action=pre_action_idx)

            obs_state_, pos_state_, distance_, done, reward, collision, pose_ = env.step(curr_action_idx,
                                                                                         goal, distance)

            if top_view:
                # top view environment used for verification of the main environment
                obsStateT, posStateT, distanceT, doneT, rewardT, collisionT, agentPoseT = envT.step(curr_action_idx,
                                                                                                    goal, distance)
            if top_view:
                # validation the postion of the agent from two diff environment_object
                train_valid_env_sync(pose_, agentPoseT)

            features_ = ae_sess.run(ae.feature_vector, feed_dict={ae.image: obs_state_[None, :, :, :]})
            features_ = np.squeeze(features_, axis=0)
            obs_pos_state_ = np.concatenate((features_, pos_state_), axis=0)

            if pose_[4] == 0:
                plt.scatter(pos_state_[0], pos_state_[2], c="white", marker="s", alpha=1)
                plt.scatter(pos_state_[0], pos_state_[2], c="red", marker="^")
            elif pose_[4] == 90:
                plt.scatter(pos_state_[0], pos_state_[2], c="white", marker="s", alpha=1)
                plt.scatter(pos_state_[0], pos_state_[2], c="red", marker=">")
            elif pose_[4] == 180:
                plt.scatter(pos_state_[0], pos_state_[2], c="white", marker="s", alpha=1)
                plt.scatter(pos_state_[0], pos_state_[2], c="red", marker="v")
            elif pose_[4] == 270:
                plt.scatter(pos_state_[0], pos_state_[2], c="white", marker="s", alpha=1)
                plt.scatter(pos_state_[0], pos_state_[2], c="red", marker="<")
            plt.pause(0.01)

            rnn_state = rnn_state_
            obs_pos_state = obs_pos_state_
            distance = distance_
            pre_action_idx = curr_action_idx
            step_num += 1

            print("\repisode:", i + 1,
                  "goal x:%2f" % goal[0], "goal z:%2f" % goal[2],
                  "x, z, :%2f" "("% pos_state_[0], "agent pos z:%2f" % pos_state_[2],
                  "distance: %3f" % distance_,
                  "step_number:", step_num)

            if done:
                if distance < distance_threshold:
                    successes += done
                else:
                    failures += done
                break
            if step_num == 200:
                done = True
                failures += 1
                break

        print("\repisode:", i + 1,
              "successes:%.3f" % (successes / (i + 1)), "failures:%.3f" % (failures / (i + 1)),
              "ratio %.3f" % (successes / (failures + 0.01)))
