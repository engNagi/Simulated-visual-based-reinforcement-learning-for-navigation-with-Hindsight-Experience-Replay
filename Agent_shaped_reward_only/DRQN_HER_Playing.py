import random
import numpy as np
import pandas as pd
import tensorflow as tf
#matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from DRQN_HER import DRQN
from Environment import Environment
from Environment_top_view import Environment_topview
from Her_episodes_experiences import her_buffer
from autoencoder import load_autoencoder
from experience_buffer import experience_buffer
from helper import train_valid_env_sync

random.seed(123)
np.random.seed(123)
dir = "/previous_Action_modified_her_with_sequence/DRQN.ckpt"

##### environment_Variables
grid_size = 0.18    # size of the agent step
top_view = True     # displaying top-view
distance_threshold = grid_size * 3  # distance threshold to the goal
action_n = 6    #number of allowed action
random_init_position = False    # Random initial positions only -- no change in the agent orientation
random_init_pose = True     # Random initial positions with random agent orientation
reward = "shaped" # reward type "shaped","sparse"

#########################   hyper-parameter
num_episodes = 50100
her_strategy = "future"
her_samples = 4
batch_size = 32
trace_length = 16
gamma = 0.99
fcl_dims = 512
nodes_num = 256
optimistion_steps = 40
epsilon_max = 1
epsilon_min = 0.05
epsilon_decay = epsilon_max - ((epsilon_max - epsilon_min) / 20000)

plotted_data = pd.DataFrame(
    columns=["Episodes", "Successful trajectories", "Failed trajectories", "Ratio", "loss", "epsilon"])


legend_elements = [Line2D([0], [0], marker="o", color="white", label="Navigable Positions",
                          markerfacecolor="grey", markersize=10),
                   Line2D([0], [0], marker="X", color="white", label="Goal Positions",
                          markerfacecolor="grey", markersize=10),
                   Line2D([0], [0], marker="o", color="white", label="Initial Agent Position",
                          markerfacecolor="blue", markersize=10),
                   Line2D([0], [0], marker="v", color="white", label="Looking Right",
                          markerfacecolor="red", markersize=10),
                   Line2D([0], [0], marker="^", color="white", label="Looking Left",
                          markerfacecolor="grey", markersize=10),
                   Line2D([0], [0], marker="<", color="white", label="looking Back",
                          markerfacecolor="red", markersize=10),
                   ]

# experience replay parameters
her_rec_buffer = her_buffer()
episode_buffer = experience_buffer(distance=distance_threshold, reward_typ=reward, her_samples=her_samples, her_strategy=her_strategy)

env = Environment(random_init_position=random_init_position, random_init_pos_orient=random_init_pose, reward_typ=reward,
                  distance=distance_threshold, random_goals=True, grid_size=grid_size, agent_mode="bot")

if top_view:
    envT = Environment_topview(grid_size=grid_size, agent_mode="bot", distance=distance_threshold, reward_typ=reward)


positions = env.get_reachable_position()
plt.ion()



#   Autoenconder
print("Autoencoder")
ae_sess, ae = load_autoencoder()

global_step = tf.Variable(0, name="global_step", trainable=False)
loss = 0

#   main loop
print("DQN_HER_Model")
drqn_graph = tf.Graph()

model = DRQN(action_n=action_n, nodes_num=nodes_num, fcl_dims=fcl_dims, scope="model",save_path=dir)
# target_model = DRQN(action_n=action_n, nodes_num=nodes_num, fcl_dims=fcl_dims, scope="target_model",
#                     save_path=dir)

with tf.Session() as sess:
    model.set_session(sess)
    # target_model.set_session(sess)
    sess.run(tf.global_variables_initializer())
    model.load()
    start = global_step.eval(sess)

    successes = 0
    failures = 0
    epsilon = 1

    for n in range(start, num_episodes):
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
        if top_view:
            # validation the position of the agent from two diff environment_object
            train_valid_env_sync(pose, pose_top)

        features = ae_sess.run(ae.feature_vector, feed_dict={ae.image: obs_state[None, :, :, :]})
        features = np.squeeze(features, axis=0)
        obs_pos_state = np.concatenate((features, pos_state), axis=0)

        plt.close()
        plt.figure()
        plt.ion()

        for pos in positions:
            plt.scatter(pos[0], pos[2], s=20, c="grey", marker="o", alpha=1)

        x_start, x_end = plt.xlim()
        y_start, y_end = plt.ylim()
        plt.xticks(np.arange(x_start, x_end, grid_size), rotation=90)
        plt.yticks(np.arange((y_start - grid_size), y_end, grid_size))
        plt.xlabel("X-Coordinates")
        plt.ylabel("Y-Coordinates")
        plt.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1), prop={"size": 6})
        plt.grid()
        plt.tight_layout()

        plt.scatter(pos_state[0], pos_state[2], c="blue", marker="o")
        plt.scatter(goal[0], goal[2], c="green", marker="X")

        plt.pause(0.9)


        if pose[4]==0:
            plt.scatter(pos_state[0], pos_state[2], c="white", marker="s", alpha=1)
            plt.scatter(pos_state[0], pos_state[2], c="red", marker="v", alpha=1)

        elif pose[4]==90:
            plt.scatter(pos_state[0], pos_state[2], c="white", marker="s", alpha=1)
            plt.scatter(pos_state[0], pos_state[2], c="red", marker=">", alpha=1)

        elif pose[4]==180:
            plt.scatter(pos_state[0], pos_state[2], c="white", marker="s", alpha=1)
            plt.scatter(pos_state[0], pos_state[2], c="red", marker="^", alpha=1)

        elif pose[4]==270:
            plt.scatter(pos_state[0], pos_state[2], c="white", marker="s", alpha=1)
            plt.scatter(pos_state[0], pos_state[2], c="red", marker="<", alpha=1)

            plt.pause(0.01)

        done = False
        while not done:

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

            if pose[4] == 0:
                plt.scatter(pos_state[0], pos_state[2], c="white", marker="s", alpha=1)
                plt.scatter(pos_state[0], pos_state[2], c="red", marker="v", alpha=1)

            elif pose[4] == 90:
                plt.scatter(pos_state[0], pos_state[2], c="white", marker="s", alpha=1)
                plt.scatter(pos_state[0], pos_state[2], c="red", marker=">", alpha=1)

            elif pose[4] == 180:
                plt.scatter(pos_state[0], pos_state[2], c="white", marker="s", alpha=1)
                plt.scatter(pos_state[0], pos_state[2], c="red", marker="^", alpha=1)

            elif pose[4] == 270:
                plt.scatter(pos_state[0], pos_state[2], c="white", marker="s", alpha=1)
                plt.scatter(pos_state[0], pos_state[2], c="red", marker="<", alpha=1)

                plt.pause(0.01)

            # # append to episode buffer
            # episode_buffer.add(np.reshape(
            #     np.array([pre_action_idx, obs_pos_state, curr_action_idx, reward, obs_pos_state_, done, goal]),
            #     [1, 7]))

            rnn_state = rnn_state_
            obs_pos_state = obs_pos_state_
            distance = distance_
            pre_action_idx = curr_action_idx

            if done:
                if distance <= distance_threshold:
                    successes += done
                else:
                    failures += done
                break

        # her_buffer = episode_buffer.her()
        # her_rec_buffer.add(her_buffer)
        # episode_buffer.clear()
        #
        # if n > 50 and n != 0:
        #     loss = model.optimize(model=model,
        #                           batch_size=batch_size,
        #                           trace_length=trace_length,
        #                           target_model=target_model,
        #                           her_buffer=her_rec_buffer,
        #                           optimization_steps=optimistion_steps)
        # if n % 100 == 0 and n > 0:
        #     print("--update model--")
        #     target_model.soft_update_from(model)
        #
        #     # model.log(drqn_summary=drqn_summary, encoder_summary=ae_summary, step=start)
        #
        # epsilon = max(epsilon * epsilon_decay, epsilon_min)

        plotted_data = plotted_data.append({"Episodes": str(n),
                                            "Successful trajectories": successes / (n + 1),
                                            "Failed trajectories": failures / (n + 1),
                                            "Ratio": (successes / (failures + 1e-6)),
                                            "loss": loss, "epsilon": epsilon}, ignore_index=True)

        #plotting_training_log(n, plotted_data, successes, failures, loss, goal, distance, pos_state, epsilon)

        # global_step.assign(n).eval()
        # #   saving
        # if n % 50 == 0 and n > 0:
        #     model.save(n)
