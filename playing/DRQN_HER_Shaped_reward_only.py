import random
import numpy as np
import tensorflow as tf

random.seed(123)
np.random.seed(123)


class DRQN(object):
    def __init__(self, action_n,
                 cell,
                 scope,
                 fcl_dims,
                 save_path,
                 input_size,
                 nodes_num,
                 gamma=0.98):

        self.cell = cell
        self.scope = scope
        self.gamma = gamma
        self.fc1_dims = fcl_dims
        self.action_n = action_n
        self.save_path = save_path
        self.nodes_num = nodes_num

        with tf.variable_scope(scope):
            # seperate agent observation, and positions
            self.inputs = tf.placeholder(tf.float32, shape=(None, 515), name="features_positions")
            # additional goals
            self.goals = tf.placeholder(tf.float32, shape=(None, 3), name="Goals_")
            # previous_action
            self.pre_action = tf.placeholder(tf.int32, shape=(None,), name="pre_action")
            # actions
            self.actions = tf.placeholder(tf.int32, shape=(None,), name="actions")
            # Q-targets-values
            self.Q_values = tf.placeholder(tf.float32, shape=(None,), name="Targets_Q_Values")

            self.pre_action_ = tf.one_hot(self.pre_action, self.action_n, dtype=tf.float32,
                                          name="pre_action_OneHot_enc")

            lstm_input = tf.concat((self.inputs, self.goals), axis=1)
            lstm_input_ = tf.concat((lstm_input, self.pre_action_), axis=1)

            with tf.variable_scope("RNN"):
                self.train_length = tf.placeholder(tf.int32)
                self.batch_size = tf.placeholder(tf.int32, shape=[])
                self.input_flat = tf.reshape(tf.layers.flatten(lstm_input_),
                                             [self.batch_size, self.train_length, input_size])

                # number_of_units may need to be changed
                # self.cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.nodes_num,
                #                                          state_is_tuple=True)

                self.state_in = self.cell.zero_state(self.batch_size, tf.float32)
                self.rnn, self.rnn_state = tf.nn.dynamic_rnn(inputs=self.input_flat,
                                                             cell=self.cell,
                                                             dtype=tf.float32,
                                                             initial_state=self.state_in,
                                                             scope=scope + '_rnn')

                self.rnn_flat = tf.reshape(self.rnn, shape=[-1, self.nodes_num])

            dense1 = tf.layers.dense(self.rnn_flat, self.fc1_dims, activation=tf.nn.relu, trainable=True)

            # final output layer
            self.predict_op = tf.layers.dense(dense1, action_n, trainable=True)

            actions_q_values = tf.reduce_sum(self.predict_op * tf.one_hot(self.actions, self.action_n),
                                             reduction_indices=[1])

            # self.clipped_Q_values = tf.clip_by_value(self.Q_values, -1 / (1 - self.gamma), 0)

            self.cost = tf.reduce_mean(tf.square(self.Q_values - actions_q_values))

            self.train_op = tf.train.AdamOptimizer(1e-3).minimize(self.cost)

            tf.summary.scalar("Cost", self.cost)
            tf.summary.histogram("Goals", self.goals)
            tf.summary.histogram("Action_Q_values", self.Q_values)
            tf.summary.histogram("LSTM", self.rnn)
            tf.summary.histogram("LSTM_State", self.rnn_state)
            self.merged = tf.summary.merge_all()

    def hard_update_from(self, other):
        mine = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        mine = sorted(mine, key=lambda v: v.name)
        theirs = [t for t in tf.trainable_variables() if t.name.startswith(other.scope)]
        theirs = sorted(theirs, key=lambda v: v.name)

        self.session.run([v_t.assign(v) for v_t, v in zip(mine, theirs)])

    def soft_update_from(self, other, tau=0.95):
        mine = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        mine = sorted(mine, key=lambda v: v.name)
        theirs = [t for t in tf.trainable_variables() if t.name.startswith(other.scope)]
        theirs = sorted(theirs, key=lambda v: v.name)

        self.session.run([v_t.assign(v_t * (1. - tau) + v * tau) for v_t, v in zip(mine, theirs)])

    def set_session(self, session):
        self.session = session

    def predict(self, pos_obs_state, goals, batch_size, trace_length, rnn_state, pre_action):
        actions_q_values, rnn, rnn_state_ = self.session.run([self.predict_op, self.rnn, self.rnn_state],
                                                             feed_dict={self.goals: goals,
                                                                        self.state_in: rnn_state,
                                                                        self.inputs: pos_obs_state,
                                                                        self.batch_size: batch_size,
                                                                        self.pre_action: pre_action,
                                                                        self.train_length: trace_length})
        return actions_q_values, rnn, rnn_state_

    def update(self, goals, states, actions, batch_size, q_values, trace_length, rnn_state, pre_action):
        self.c, _ = self.session.run([self.cost, self.train_op],
                                     feed_dict={self.goals: goals,
                                                self.inputs: states,
                                                self.actions: actions,
                                                self.Q_values: q_values,
                                                self.state_in: rnn_state,
                                                self.batch_size: batch_size,
                                                self.pre_action: pre_action,
                                                self.train_length: trace_length})
        return self.c

    def sample_action(self, goal, batch_size, trace_length, epsilon, rnn_state, pos_obs_state, pre_action):
        """Implements epsilon greedy algorithm"""
        if np.random.random() < epsilon:
            q_values, rnn, rnn_state_ = self.predict(pos_obs_state=[pos_obs_state],
                                                     goals=[goal],
                                                     pre_action=[pre_action],
                                                     batch_size=batch_size,
                                                     trace_length=trace_length,
                                                     rnn_state=rnn_state)
            action = np.random.randint(1, self.action_n)
        else:
            action_q_values, _, rnn_state_ = self.predict(pos_obs_state=[pos_obs_state],
                                                          goals=[goal],
                                                          pre_action=[pre_action],
                                                          batch_size=batch_size,
                                                          trace_length=trace_length,
                                                          rnn_state=rnn_state)
            action = np.argmax(action_q_values[0])
        return action, rnn_state_

    def load(self):
        self.saver = tf.train.Saver(tf.global_variables())
        load_was_success = True
        try:
            save_dir = '/'.join(self.save_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(save_dir)
            load_path = ckpt.model_checkpoint_path
            self.saver.restore(self.session, load_path)
        except:
            print("no saved model to load. starting new session")
            load_was_success = False
        else:
            print("loaded model: {}".format(load_path))
            saver = tf.train.Saver(tf.global_variables())
            episode_number = int(load_path.split('-')[-1])

    def save(self, n):
        self.saver.save(self.session, self.save_path, global_step=n)
        print("SAVED MODEL #{}".format(n))

    def optimize(self, model, target_model, batch_size, trace_length, her_buffer, optimization_steps):
        losses = 0

        for _ in range(optimization_steps):
            rnn_stat_train = (np.zeros([batch_size, self.nodes_num]), np.zeros([batch_size, self.nodes_num]))

            train_batch = her_buffer.sample(batch_size=batch_size, trace_length=trace_length)

            pre_action, states, curr_actions, rewards, next_states, dones, goals = map(np.array, zip(*train_batch))
            # Calculate targets
            next_Qs, _, _ = target_model.predict(goals=goals,
                                                 pre_action=pre_action,
                                                 pos_obs_state=next_states,
                                                 rnn_state=rnn_stat_train,
                                                 trace_length=trace_length,
                                                 batch_size=batch_size)
            next_Q = np.amax(next_Qs, axis=1)
            target_q_values = rewards + np.invert(dones).astype(np.float32) * self.gamma * next_Q
            #   Calculate network loss
            loss = model.update(goals=goals,
                                states=states,
                                actions=curr_actions,
                                pre_action=pre_action,
                                rnn_state=rnn_stat_train,
                                q_values=target_q_values,
                                trace_length=trace_length,
                                batch_size=batch_size)
            losses += loss
        return losses / optimization_steps

    def log(self, encoder_summary, drqn_summary, step):
        encoder_writer = tf.summary.FileWriter("/home/nagi/Desktop/Master_Project/DRQN_features_pos_2/encoder")
        encoder_writer.add_summary(encoder_summary, global_step=step)
        writer = tf.summary.FileWriter("/home/nagi/Desktop/Master_Project/DRQN_features_pos_2/Train")
        writer.add_summary(drqn_summary, global_step=step)
