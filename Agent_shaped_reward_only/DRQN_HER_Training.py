import random
import numpy as np
import pandas as pd
import tensorflow as tf
from DRQN_HER import DRQN
from Environment import Environment
from Environment_top_view import Environment_topview
from Her_episodes_experiences import her_buffer
from autoencoder import load_autoencoder
from experience_buffer import experience_buffer
from helper import plotting_training_log, train_valid_env_sync, validate

random.seed(123)
np.random.seed(123)

dir = "/home/nagi/Desktop/Master_project_final/DRQN_3_shaped_reward_only_sequence/DRQN.ckpt"

##### environment_Variables
grid_size = 0.18  # size of the agent step
top_view = True  # displaying top-view
distance_threshold = grid_size * 2  # distance threshold to the goal
action_n = 3  # number of allowed action
random_init_position = False  # Random initial positions only -- no change in the agent orientation
random_init_pose = True  # Random initial positions with random agent orientation
reward = "shaped"  # reward type "shaped","sparse"

#########################   hyper-parameter
num_episodes = 15001
her_samples = 8
batch_size = 32
trace_length = 8
gamma = 0.99
fcl_dims = 512
nodes_num = 256
optimistion_steps = 40
epsilon_max = 1
epsilon_min = 0.001
input_size = 521  ## size of the input to the LSTM
epsilon_decay = epsilon_max - ((epsilon_max / 3500))


## pandas data-frame for plotting
plotted_data = pd.DataFrame(
    columns=["Episodes", "Successful trajectories", "Failed trajectories", "Ratio", "loss", "epsilon"])

# experience replay parameters
her_rec_buffer = her_buffer()
episode_buffer = experience_buffer(distance=distance_threshold, reward_typ=reward, her_samples=her_samples)

env = Environment(random_init_position=random_init_position, random_init_pos_orient=random_init_pose, reward_typ=reward,
                  distance=distance_threshold, random_goals=True, grid_size=grid_size, agent_mode="bot")

if top_view:
    envT = Environment_topview(grid_size=grid_size, agent_mode="bot", distance=distance_threshold, reward_typ=reward)

#   Autoenconder
print("Autoencoder")
ae_sess, ae = load_autoencoder()

global_step = tf.Variable(0, name="global_step", trainable=False)
loss = 0

#   main loop
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
    start = global_step.eval(sess)

    successes = 0
    failures = 0
    epsilon = 1

    for n in range(start, num_episodes):
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
        if top_view:
            # validation the position of the agent from two diff environment_object
            train_valid_env_sync(pose, pose_top)

        features = ae_sess.run(ae.feature_vector, feed_dict={ae.image: obs_state[None, :, :, :]})
        features = np.squeeze(features, axis=0)
        obs_pos_state = np.concatenate((features, pos_state), axis=0)

        done = False
        while not done:

            curr_action_idx, rnn_state_ = model.sample_action(goal=goal,
                                                              batch_size=1,
                                                              trace_length=1,
                                                              epsilon=epsilon,
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

            # append to episode buffer
            episode_buffer.add(np.reshape(
                np.array([pre_action_idx, obs_pos_state, curr_action_idx, reward, obs_pos_state_, done, goal]),
                [1, 7]))

            rnn_state = rnn_state_
            obs_pos_state = obs_pos_state_
            distance = distance_
            pre_action_idx = curr_action_idx
            step_num += 1
            if done:
                if distance < distance_threshold:
                    successes += done
                else:
                    failures += done
                break
            if step_num == 200:
                    done = True
                    failures += done
                    break

        #her_buffer = episode_buffer.her()
        her_rec_buffer.add([episode_buffer.memory])
        episode_buffer.clear()

        plotted_data = plotted_data.append({"Episodes": str(n),
                                            "Successful trajectories": successes / (n + 1),
                                            "Failed trajectories": failures / (n + 1),
                                            "Ratio": (successes / (failures + 1)),
                                            "loss": loss, "epsilon": epsilon,
                                            "F1": ((1-(failures / (n + 1))) * (successes / ( n + 1))) /
                                                  (((1-(failures / (n + 1))) + ((successes / ( n + 1))))+1)}, ignore_index=True)


        plotting_training_log(n, plotted_data, successes, failures, loss, goal, distance, pos_state, epsilon, step_num)

        ###validation###
        if n % 4000 == 0 and n > 0:
            validate(n=n, nodes_num=nodes_num, top_view=top_view, env=env, envT=envT, ae=ae, ae_sess=ae_sess,
                     distance_threshold=distance_threshold, model=model)

        if n > 50 and n != 0:
            loss = model.optimize(model=model,
                                  batch_size=batch_size,
                                  trace_length=trace_length,
                                  target_model=target_model,
                                  her_buffer=her_rec_buffer,
                                  optimization_steps=optimistion_steps)
        if n % 4000 == 0 and n > 0:
            print("#### update model ####")
            target_model.soft_update_from(model)

        # model.log(drqn_summary=drqn_summary, encoder_summary=ae_summary, step=start)

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        global_step.assign(n).eval()
        #   saving
        if n % 50 == 0 and n > 0:
            model.save(n)
