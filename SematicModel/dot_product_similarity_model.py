import tensorflow.contrib.layers as layers
import tensorflow as tf
import numpy as np

class DotProductSimilarityModel():
    def __init__(self, n_grams_size, embedding_size, batch_size, is_training, learning_rate, device='/cpu:0',
                 scope=None):
        self.n_grams_size = n_grams_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        with tf.variable_scope(scope or 'tcm') as scope:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            if is_training is not None:
                self.is_training = is_training
            else:
                self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')
            self.x_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='x_inputs')
            self.y_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='y_inputs')

            self._init_embedding(scope)
            with tf.device(device):
                self._init_body(scope)
        with tf.variable_scope('train'):
            self.loss = tf.divide(tf.reduce_sum(tf.log(tf.reduce_sum(tf.exp(self.similarity), 1))) - tf.cast(
                tf.reduce_sum(tf.diag_part(self.similarity)), tf.float32), tf.cast(self.batch_size, tf.float32))
            tf.summary.scalar('loss', self.loss)

            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

            self.summary_op = tf.summary.merge_all()


    def _init_embedding(self, scope):
        with tf.variable_scope(scope):
            with tf.variable_scope("embedding") as scope:
                self.embedding_matrix = tf.get_variable(
                    name="embedding_matrix",
                    shape=[self.n_grams_size, self.embedding_size],
                    initializer=layers.xavier_initializer(),
                    dtype=tf.float32, trainable=False)

                x_inputs_sum_embedded = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)
                y_inputs_sum_embedded = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)

                def cond(i, x_input0, y_input0, x_inputs_sum_embedded, y_inputs_sum_embedded):
                    return tf.less(i, tf.shape(x_input0)[0])

                def body(i, x_input0, y_input0, x_inputs_sum_embedded, y_inputs_sum_embedded):
                    def cond1(j, input1, temp):
                        return tf.less(j, tf.shape(input1)[0])

                    def body1(j, input1, temp):
                        def execution_true():
                            return tf.nn.embedding_lookup(self.embedding_matrix, input1[j])

                        def execution_false():
                            return tf.constant(0.0, shape=[self.embedding_size])

                        temp = temp.write(j, tf.cond(tf.greater(input1[j], tf.cast(tf.constant(-1), tf.int32)),
                                                     execution_true, execution_false))
                        j = j + 1
                        return j, input1, temp

                    j = tf.constant(0)
                    temp = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)
                    j, x_input0_i, temp = tf.while_loop(cond1, body1, [j, x_input0[i], temp])
                    x_inputs_sum_embedded = x_inputs_sum_embedded.write(i, tf.reduce_sum(temp.stack(), 0))

                    j = tf.constant(0)
                    temp = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)
                    j, y_input0_i, temp = tf.while_loop(cond1, body1, [j, y_input0[i], temp])
                    y_inputs_sum_embedded = y_inputs_sum_embedded.write(i, tf.reduce_sum(temp.stack(), 0))

                    i = i + 1
                    return i, x_input0, y_input0, x_inputs_sum_embedded, y_inputs_sum_embedded

                i = tf.constant(0)
                i, a, b, x_inputs_sum_embedded, y_inputs_sum_embedded = tf.while_loop(cond, body,
                                                                                      [i, self.x_inputs, self.y_inputs,
                                                                                       x_inputs_sum_embedded,
                                                                                       y_inputs_sum_embedded])
                self.final_x_inputs_sum_embedded = x_inputs_sum_embedded.stack()
                self.final_y_inputs_sum_embedded = y_inputs_sum_embedded.stack()

    def _init_body(self, scope):
        with tf.variable_scope(scope):
            x_w_1 = tf.get_variable(name="x_w_1", shape=[320, 300], dtype=tf.float32,
                                    initializer=layers.xavier_initializer())
            x_w_2 = tf.get_variable(name="x_w_2", shape=[300, 300], dtype=tf.float32,
                                    initializer=layers.xavier_initializer())
            x_w_3 = tf.get_variable(name="x_w_3", shape=[300, 500], dtype=tf.float32,
                                    initializer=layers.xavier_initializer())

            x_b_1 = tf.get_variable(name="x_b_1", shape=[1, 300], dtype=tf.float32, initializer=tf.zeros_initializer())
            x_b_2 = tf.get_variable(name="x_b_2", shape=[1, 300], dtype=tf.float32, initializer=tf.zeros_initializer())
            x_b_3 = tf.get_variable(name="x_b_3", shape=[1, 500], dtype=tf.float32, initializer=tf.zeros_initializer())

            y_w_1 = tf.get_variable(name="y_w_1", shape=[320, 300], dtype=tf.float32,
                                    initializer=layers.xavier_initializer())
            y_w_2 = tf.get_variable(name="y_w_2", shape=[300, 300], dtype=tf.float32,
                                    initializer=layers.xavier_initializer())
            y_w_3 = tf.get_variable(name="y_w_3", shape=[300, 500], dtype=tf.float32,
                                    initializer=layers.xavier_initializer())

            y_b_1 = tf.get_variable(name="y_b_1", shape=[1, 300], dtype=tf.float32, initializer=tf.zeros_initializer())
            y_b_2 = tf.get_variable(name="y_b_2", shape=[1, 300], dtype=tf.float32, initializer=tf.zeros_initializer())
            y_b_3 = tf.get_variable(name="y_b_3", shape=[1, 500], dtype=tf.float32, initializer=tf.zeros_initializer())

            x_fc1 = tf.nn.tanh(tf.matmul(self.final_x_inputs_sum_embedded, x_w_1) + x_b_1)  # (none,300)
            x_fc2 = tf.nn.tanh(tf.matmul(x_fc1, x_w_2) + x_b_2)  # (none,300)
            x_fc3 = tf.nn.tanh(tf.matmul(x_fc2, x_w_3) + x_b_3)  # (none,500)

            y_fc1 = tf.nn.tanh(tf.matmul(self.final_y_inputs_sum_embedded, y_w_1) + y_b_1)  # (none,300)
            y_fc2 = tf.nn.tanh(tf.matmul(y_fc1, y_w_2) + y_b_2)  # (none,300)
            y_fc3 = tf.nn.tanh(tf.matmul(y_fc2, y_w_3) + y_b_3)  # (none,500)

            self.similarity = tf.tensordot(x_fc3, y_fc3, axes=([1], [1]))  # (none,none)



    def get_feed_data(self, X, Y, is_training=True):

        def padding(inputs):
            batch_size = len(inputs)
            n_grams_sizes = np.array([len(doc) for doc in inputs], dtype=np.int32)
            max_n_grams_sizes = n_grams_sizes.max()

            inputs_padded = np.full((batch_size, max_n_grams_sizes), -1)
            for i, document in enumerate(inputs):
                for j, feature in enumerate(document):
                    inputs_padded[i, j] = feature
            return inputs_padded

        fd = {
            self.x_inputs: np.array(padding(X)),
            self.y_inputs: np.array(padding(Y))
        }
        fd[self.is_training] = is_training

        return fd
