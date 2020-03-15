import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.lines import Line2D


# plt.style.use("seaborn")


def train_valid_env_sync(training_env_pose, validation_env_pose):
    equal = np.array_equal(training_env_pose, training_env_pose)
    if not equal:
        print("Agent position x:", training_env_pose[0], "not equal top view position x:", validation_env_pose[0],
              "Agent position z:", training_env_pose[2], "not equal top view position z:", validation_env_pose[2],
              "Agent angle :", training_env_pose[4], "not equal top view angle", validation_env_pose[4])


def plotting_training_log(num_episode, plotted_data, successes, failures, loss, goal, distance, agent_init_pos,
                          epsilon):
    print("\repisode:", num_episode + 1,
          "successes:%.3f" % (successes / (num_episode + 1)),
          "goal x:%2f" % goal[0], "goal z:%2f" % goal[2],
          "agent pos x:%2f" % agent_init_pos[0], "agent pos z:%2f" % agent_init_pos[2],
          "distance: %3f" % distance,
          "failures:%.3f" % (failures / (num_episode + 1)),
          "ratio %.3f" % (successes / (failures + 1e-6)),
          "loss: %.2f" % loss, "exploration %.5f" % epsilon)

    if num_episode % 100 == 0 and num_episode > 0:
        #   combined plot of successful failed trajectories and Ratio between them
        plotted_data.plot(x="Episodes", y=["Successful trajectories", "Failed trajectories", "Ratio"],
                          title="Agent Learning Ratio")
        plt.xlabel("Episodes")
        plt.ylabel("Successful/Failed Trajectories and Ratio")
        plt.savefig("failed_success_ratio" + str(num_episode) + ".png")

        #   plot of successful trajectories
        plotted_data.plot(x="Episodes", y=["Successful trajectories"],
                          title="Successful Trajectories")
        plt.xlabel("Episodes")
        plt.ylabel("Successful Trajectories")
        plt.savefig("successful" + str(num_episode) + ".png")

        #   plot of failed trajectories
        plotted_data.plot(x="Episodes", y=["Failed trajectories"],
                          title="Failed Trajectories")
        plt.xlabel("Episodes")
        plt.ylabel("Failed Trajectories")
        plt.savefig("Failed" + str(num_episode) + ".png")

        plotted_data.plot(x="Episodes", y=["Ratio"],
                          title="Ratio between successful and failed trajectories")
        plt.xlabel("Episodes")
        plt.ylabel("Ratio")
        plt.savefig("Ratio" + str(num_episode) + ".png")

        plotted_data.plot(x="Episodes", y=["loss"],
                          title="HER-DRQN model loss")
        plt.xlabel("Episodes")
        plt.ylabel("Loss")
        plt.savefig("Loss" + str(num_episode) + ".png")


def validate(n, nodes_num, top_view, env, envT, ae, ae_sess, distance_threshold, model):
    print("### Validation ###")
    plotted_data_val = pd.DataFrame(
        columns=["Episodes", "Successful trajectories", "Failed trajectories", "Ratio", "loss", "epsilon", "num_steps"])
    val_success = 0
    val_failures = 0
    for i in range(100):
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
        num_steps = 0
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
                obsStateT, posStateT, distanceT, doneT, rewardT, collisionT, agentPoseT = envT.step(
                    curr_action_idx,
                    goal, distance)
            if top_view:
                # validation the postion of the agent from two diff environment_object
                train_valid_env_sync(pose_, agentPoseT)

            features_ = ae_sess.run(ae.feature_vector, feed_dict={ae.image: obs_state_[None, :, :, :]})
            features_ = np.squeeze(features_, axis=0)
            obs_pos_state_ = np.concatenate((features_, pos_state_), axis=0)

            rnn_state = rnn_state_
            obs_pos_state = obs_pos_state_
            distance = distance_
            pre_action_idx = curr_action_idx
            num_steps += 1
            if done:
                if distance <= distance_threshold:
                    val_success += done
                else:
                    val_failures += done
            if num_steps == 150:
                done = True
                val_failures += done
                break

        print("validation_success:", val_success, "validation_failures:", val_failures, "steps_num",num_steps)
        plotted_data_val = plotted_data_val.append({"Episodes": str(i),
                                                    "Successes": val_success / (i+1),
                                                    "Failures": val_failures / (i+1),
                                                    "Ratio": (val_success / (val_failures + 0.1)),
                                                    "num_steps": num_steps}, ignore_index=True)

    plotted_data_val.plot(x="Episodes", y=["Successes", "Failures", "Ratio"],
                          title="Validation Agent Learning Ratio")
    plt.xlabel("Episodes")
    plt.ylabel("Successful/Failed Trajectories and Ratio")
    plt.savefig("Vaildation_failed_success_ratio " +str(n) + str(i) + ".png")
