import tensorflow as tf
import numpy as np
import glob, random, os

tf.logging.set_verbosity(tf.logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model_path = "saved_models/"
model_name = model_path + 'model'
training_dataset_path = "/home/WIN-UNI-DUE/sjmonagi/Desktop/Master_Project/Dataset_2/training"
validation_dataset_path = "/home/WIN-UNI-DUE/sjmonagi/Desktop/Master_Project/Dataset_2/validation"
training = 40000
validation_iter = 5000
total_t = 0
validation_period = 5000


class Network(object):
    # Create model
    def __init__(self):
        self.save_path = '/home/nagi/Desktop/Master_project_final/autoencoder_add_2/CNN_AE.ckpt'
        self.image = tf.placeholder(tf.float32, [None, 300, 300, 3], name='image')
        self.resized_image = tf.image.resize_images(self.image, [256, 256])
        self.normalized_image = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), self.resized_image)

        # self.normalized_image = tf.image.per_image_standardization([self.resized_image[1],
        #                                                             self.resized_image[2],
        #                                                             self.resized_image[3]])

        self.feature_vector = self.encoder(self.normalized_image)
        self.reconstructions = self.decoder(self.feature_vector)

        self.loss = self.compute_loss()
        optimizer = tf.train.AdamOptimizer(1e-3)

        with tf.variable_scope("gradient_clip"):
            gradients = optimizer.compute_gradients(self.loss)
            clipped_grad = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]

        self.train_op = optimizer.apply_gradients(clipped_grad)
        #   summaries
        tf.summary.image('resized', self.normalized_image, 20)
        tf.summary.image('reconstructed_normalized_image', self.reconstructions, 20)
        tf.summary.histogram("reconstructed", self.reconstructions)
        tf.summary.scalar('loss', self.loss)

        self.merged = tf.summary.merge_all()

    def encoder(self, x):
        with tf.variable_scope("Decoder"):
            conv_1 = tf.layers.conv2d(x, filters=4, kernel_size=3, strides=2, padding='same',
                                      activation=tf.nn.leaky_relu,
                                      kernel_initializer=tf.initializers.he_normal(),
                                      name="Conv_1")

            conv_1_1 = tf.layers.conv2d(conv_1, filters=4, kernel_size=3, strides=1, padding='same',
                                        activation=tf.nn.leaky_relu,
                                        kernel_initializer=tf.initializers.he_normal(),
                                        name="Conv_1_1")
            conv_1_2 = tf.layers.conv2d(conv_1_1, filters=4, kernel_size=3, strides=1, padding='same',
                                        activation=tf.nn.leaky_relu,
                                        kernel_initializer=tf.initializers.he_normal(),
                                        name="Conv_1_2")
            conv_1_add = tf.add(conv_1, conv_1_2, name="Conv_1_add")

            conv_2 = tf.layers.conv2d(conv_1_add, filters=8, kernel_size=3, strides=2, padding='same',
                                      activation=tf.nn.leaky_relu,
                                      kernel_initializer=tf.initializers.he_normal(),
                                      name="Conv_2")
            conv_2_1 = tf.layers.conv2d(conv_2, filters=8, kernel_size=3, strides=1, padding='same',
                                        activation=tf.nn.leaky_relu,
                                        kernel_initializer=tf.initializers.he_normal(),
                                        name="Conv_2_1")
            conv_2_2 = tf.layers.conv2d(conv_2_1, filters=8, kernel_size=3, strides=1, padding='same',
                                        activation=tf.nn.leaky_relu,
                                        kernel_initializer=tf.initializers.he_normal(),
                                        name="Conv_2_2")
            conv_2_add = tf.add(conv_2, conv_2_2, name="Conv_2_add")

            conv_3 = tf.layers.conv2d(conv_2_add, filters=16, kernel_size=3, strides=2, padding='same',
                                      activation=tf.nn.leaky_relu,
                                      kernel_initializer=tf.initializers.he_normal(),
                                      name="Conv_3")
            conv_3_1 = tf.layers.conv2d(conv_3, filters=16, kernel_size=3, strides=1, padding='same',
                                        activation=tf.nn.leaky_relu,
                                        kernel_initializer=tf.initializers.he_normal(),
                                        name="Conv_3_1")
            conv_3_2 = tf.layers.conv2d(conv_3_1, filters=16, kernel_size=3, strides=1, padding='same',
                                        activation=tf.nn.leaky_relu,
                                        kernel_initializer=tf.initializers.he_normal(),
                                        name="Conv_3_2")
            conv_3_add = tf.add(conv_3, conv_3_2, name="Conv_3_add")

            conv_4 = tf.layers.conv2d(conv_3_add, filters=32, kernel_size=3, strides=2, padding='same',
                                      activation=tf.nn.leaky_relu,
                                      kernel_initializer=tf.initializers.he_normal(),
                                      name="Conv_4")
            conv_4_1 = tf.layers.conv2d(conv_4, filters=32, kernel_size=3, strides=1, padding='same',
                                        activation=tf.nn.leaky_relu,
                                        kernel_initializer=tf.initializers.he_normal(),
                                        name="Conv_4_1")
            conv_4_2 = tf.layers.conv2d(conv_4_1, filters=32, kernel_size=3, strides=1, padding='same',
                                        activation=tf.nn.leaky_relu,
                                        kernel_initializer=tf.initializers.he_normal(),
                                        name="Conv_4_2")
            conv_4_add = tf.add(conv_4, conv_4_2, name="Conv_4_add")

            conv_5 = tf.layers.conv2d(conv_4_add, filters=64, kernel_size=3, strides=2, padding='same',
                                      activation=tf.nn.leaky_relu,
                                      kernel_initializer=tf.initializers.he_normal(),
                                      name="Conv_5")
            conv_5_1 = tf.layers.conv2d(conv_5, filters=64, kernel_size=3, strides=1, padding='same',
                                        activation=tf.nn.leaky_relu,
                                        kernel_initializer=tf.initializers.he_normal(),
                                        name="Conv_5_1")
            conv_5_2 = tf.layers.conv2d(conv_5_1, filters=64, kernel_size=3, strides=1, padding='same',
                                        activation=tf.nn.leaky_relu,
                                        kernel_initializer=tf.initializers.he_normal(),
                                        name="Conv_5_2")
            conv_5_add = tf.add(conv_5, conv_5_2, name="Conv_5_add")

            conv_6 = tf.layers.conv2d(conv_5_add, filters=128, kernel_size=3, strides=1, padding='same',
                                      activation=tf.nn.leaky_relu,
                                      kernel_initializer=tf.initializers.he_normal(),
                                      name="Conv_6")
            conv_6_1 = tf.layers.conv2d(conv_6, filters=128, kernel_size=3, strides=1, padding='same',
                                        activation=tf.nn.leaky_relu,
                                        kernel_initializer=tf.initializers.he_normal(),
                                        name="Conv_6_1")

            conv_6_2 = tf.layers.conv2d(conv_6_1, filters=128, kernel_size=3, strides=1, padding='same',
                                        activation=tf.nn.leaky_relu,
                                        kernel_initializer=tf.initializers.he_normal(),
                                        name="Conv_6_2")
            conv_6_add = tf.add(conv_6, conv_6_2, name="Conv_6_add")

            x = tf.layers.flatten(conv_6_add)
            z = tf.layers.dense(x, units=512, name='z_mu')
        return z

    def decoder(self, z):
        with tf.variable_scope("Encoder"):
            x = tf.layers.dense(z, 1024, activation=tf.nn.leaky_relu,
                                kernel_initializer=tf.initializers.he_normal())  # 65536
            x = tf.reshape(x, [-1, 8, 8, 16])
            conv_8 = tf.layers.conv2d_transpose(x, filters=256, kernel_size=3, strides=2, padding='same',
                                                activation=tf.nn.leaky_relu,
                                                kernel_initializer=tf.initializers.he_normal(),
                                                name="Conv_8")
            conv_8_1 = tf.layers.conv2d_transpose(conv_8, filters=256, kernel_size=3, strides=1, padding='same',
                                                  activation=tf.nn.leaky_relu,
                                                  kernel_initializer=tf.initializers.he_normal(),
                                                  name="Conv_8_1")
            conv_8_2 = tf.layers.conv2d_transpose(conv_8_1, filters=256, kernel_size=3, strides=1, padding='same',
                                                  activation=tf.nn.leaky_relu,
                                                  kernel_initializer=tf.initializers.he_normal(),
                                                  name="Conv_8_2")
            conv_8_add = tf.add(conv_8, conv_8_2, name="Conv_8_add")

            conv_9 = tf.layers.conv2d_transpose(conv_8_add, filters=128, kernel_size=3, strides=2, padding='same',
                                                activation=tf.nn.leaky_relu,
                                                kernel_initializer=tf.initializers.he_normal(),
                                                name="Conv_9")
            conv_9_1 = tf.layers.conv2d_transpose(conv_9, filters=128, kernel_size=3, strides=1, padding='same',
                                                  activation=tf.nn.leaky_relu,
                                                  kernel_initializer=tf.initializers.he_normal(),
                                                  name="Conv_9_1")
            conv_9_2 = tf.layers.conv2d_transpose(conv_9_1, filters=128, kernel_size=3, strides=1, padding='same',
                                                  activation=tf.nn.leaky_relu,
                                                  kernel_initializer=tf.initializers.he_normal(),
                                                  name="Conv_9_2")
            conv_9_add = tf.add(conv_9, conv_9_2, name="Conv_9_add")

            conv_10 = tf.layers.conv2d_transpose(conv_9_add, filters=64, kernel_size=3, strides=2, padding='same',
                                                 activation=tf.nn.leaky_relu,
                                                 kernel_initializer=tf.initializers.he_normal(),
                                                 name="Conv_10")

            conv_10_1 = tf.layers.conv2d_transpose(conv_10, filters=64, kernel_size=3, strides=1, padding='same',
                                                   activation=tf.nn.leaky_relu,
                                                   kernel_initializer=tf.initializers.he_normal(),
                                                   name="Conv_10_1")
            conv_10_2 = tf.layers.conv2d_transpose(conv_10_1, filters=64, kernel_size=3, strides=1, padding='same',
                                                   activation=tf.nn.leaky_relu,
                                                   kernel_initializer=tf.initializers.he_normal(),
                                                   name="Conv_10_2")
            conv_10_add = tf.add(conv_10, conv_10_2, name="Conv_10_add")

            conv_11 = tf.layers.conv2d_transpose(conv_10_add, filters=32, kernel_size=3, strides=2, padding='same',
                                                 activation=tf.nn.leaky_relu,
                                                 kernel_initializer=tf.initializers.he_normal(),
                                                 name="Conv_11")
            conv_11_1 = tf.layers.conv2d_transpose(conv_11, filters=32, kernel_size=3, strides=1, padding='same',
                                                   activation=tf.nn.leaky_relu,
                                                   kernel_initializer=tf.initializers.he_normal(),
                                                   name="Conv_11_1")
            conv_11_2 = tf.layers.conv2d_transpose(conv_11_1, filters=32, kernel_size=3, strides=1, padding='same',
                                                   activation=tf.nn.leaky_relu,
                                                   kernel_initializer=tf.initializers.he_normal(),
                                                   name="Conv_11_2")
            conv_11_add = tf.add(conv_11, conv_11_2, name="Conv_11_add")

            conv_12 = tf.layers.conv2d_transpose(conv_11_add, filters=16, kernel_size=3, strides=2, padding='same',
                                                 activation=tf.nn.leaky_relu,
                                                 kernel_initializer=tf.initializers.he_normal(),
                                                 name="Conv_12")
            conv_12_1 = tf.layers.conv2d_transpose(conv_12, filters=16, kernel_size=3, strides=1, padding='same',
                                                   activation=tf.nn.leaky_relu,
                                                   kernel_initializer=tf.initializers.he_normal(),
                                                   name="Conv_12_1")
            conv_12_2 = tf.layers.conv2d_transpose(conv_12_1, filters=16, kernel_size=3, strides=1, padding='same',
                                                   activation=tf.nn.leaky_relu,
                                                   kernel_initializer=tf.initializers.he_normal(),
                                                   name="Conv_12_2")
            conv_12_add = tf.add(conv_12, conv_12_2, name="Conv_12_add")

            conv_13 = tf.layers.conv2d_transpose(conv_12_add, filters=3, kernel_size=3, strides=1, padding='same',
                                                 activation=tf.nn.leaky_relu,
                                                 kernel_initializer=tf.initializers.he_normal(),
                                                 name="Conv_13")
            conv_13_1 = tf.layers.conv2d_transpose(conv_13, filters=3, kernel_size=3, strides=1, padding='same',
                                                   activation=tf.nn.leaky_relu,
                                                   kernel_initializer=tf.initializers.he_normal(),
                                                   name="Conv_13_1")
            conv_13_2 = tf.layers.conv2d_transpose(conv_13_1, filters=3, kernel_size=3, strides=1, padding='same',
                                                   activation=None,
                                                   kernel_initializer=tf.initializers.he_normal(),
                                                   name="Conv_13_2")
            conv_13_add = tf.add(conv_13, conv_13_2, name="Conv_13_add")

        return conv_13_add

    def compute_loss(self):
        with tf.variable_scope("RMS_Loss"):
            batch_shape = tf.shape(self.normalized_image)[0]
            logits_flat = tf.reshape(self.reconstructions, [batch_shape, -1])
            labels_flat = tf.reshape(self.normalized_image, [batch_shape, -1])
            reconstruction_loss = tf.reduce_sum(tf.square(logits_flat - labels_flat), axis=1)
            vae_loss = tf.reduce_mean(reconstruction_loss)

        return vae_loss

    def set_session(self, session):
        self.session = session

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


def data_iterator(batch_size, path):
    data_files = glob.glob(path + '/**/VAE_FloorPlan*', recursive=True)
    while True:
        data = np.load(random.sample(data_files, 1)[0])
        np.random.shuffle(data)
        np.random.shuffle(data)
        N = data.shape[0]
        start = np.random.randint(0, N - batch_size)
        yield data[start:start + batch_size]


def train_vae():
    network = Network()
    with tf.Session() as sess:
        network.set_session(session=sess)
        sess.run(tf.global_variables_initializer())
        network.load()

        loss_writer = tf.summary.FileWriter('Autoencoder_2_add/train', sess.graph)
        validation_writer = tf.summary.FileWriter('Autoencoder_2_add/validation')

        training_data = data_iterator(batch_size=32, path=training_dataset_path)
        validation_data = data_iterator(batch_size=16, path=validation_dataset_path)

        for i in range(training):
            training_images = next(training_data)
            training_loss, _, loss_summary = sess.run([network.loss, network.train_op, network.merged],
                                                      feed_dict={network.image: training_images})
            print('step {}: training loss {:.6f}'.format(i, training_loss))
            loss_writer.add_summary(loss_summary)

            # print("normalized_input", network.normalized_image.eval(feed_dict={network.image: training_images}))
            # print("reconstructed", network.reconstructions.eval(feed_dict={network.image: training_images}))
            # print("input",network.image.eval(feed_dict={network.image: training_images}))

            if i % 10000 == 0 and i > 0:  # validation
                print("validation")
                for i in range(validation_iter):
                    validation_images = next(validation_data)
                    validation_loss, validation_summary = sess.run([network.loss, network.merged],
                                                                   feed_dict={network.image: validation_images})
                    validation_writer.add_summary(validation_summary)
                    print('step {}: validation loss {:.6f}'.format(i, validation_loss))

            if i % 50 == 0:
                network.save(i)


def load_autoencoder():
    AE_CNN = tf.Graph()
    with AE_CNN.as_default():
        network = Network()
    AE_CNN_sess = tf.Session(graph=AE_CNN)
    network.set_session(AE_CNN_sess)
    with AE_CNN_sess.as_default():
        with AE_CNN.as_default():
            tf.global_variables_initializer().run()
            network.load()

        return AE_CNN_sess, network


if __name__ == "__main__":
    train_vae()
