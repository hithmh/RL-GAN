import tensorflow as tf
import numpy as np
import os
import shutil


smooth = 0
def doom_feature_abstraction(s):
    init_w = tf.contrib.layers.xavier_initializer()
    init_b = tf.constant_initializer(0.001)
    with tf.variable_scope('feature_abstraction'):
        s = tf.layers.conv2d(s, 8, [5, 5], [4, 4], name='l1', activation=tf.nn.elu,
                             kernel_constraint=init_w, bias_initializer=init_b)
        s = tf.layers.conv2d(s, 16, [3, 3], [2, 2], name='l2', activation=tf.nn.elu,
                             kernel_constraint=init_w, bias_initializer=init_b)
        s = tf.layers.conv2d(s, 32, [3, 3], [2, 2], name='l3', activation=tf.nn.elu,
                             kernel_constraint=init_w, bias_initializer=init_b)
        s = tf.layers.conv2d(s, 64, [3, 3], [2, 2], name='l4', activation=tf.nn.elu,
                             kernel_constraint=init_w, bias_initializer=init_b)
        s = tf.reshape(s, [-1, np.prod(s.get_shape().as_list()[1:])])
        s = tf.layers.dense(s, 256, activation= tf.nn.elu, name='fc', kernel_constraint=init_w, bias_initializer=init_b)
    return s

def universe_feature_abstraction(s,nConvs=4, reuse = False):
    print('Using universe head design')
    init_w = tf.contrib.layers.xavier_initializer()
    init_b = tf.constant_initializer(0.001)
    with tf.variable_scope('feature_abstraction', reuse = reuse):
        for i in range(nConvs):

            s = tf.layers.conv2d(s, 32, [3, 3], [2, 2], name="l{}".format(i + 1), activation=tf.nn.elu,
                                 kernel_constraint=init_w, bias_initializer=init_b)
            # print('Loop{} '.format(i+1),tf.shape(x))
            # print('Loop{}'.format(i+1),x.get_shape())
        s = tf.reshape(s, [-1, np.prod(s.get_shape().as_list()[1:])])
    return s

def build_LSTM(phis, n_lstm_unit = 256, time_step =4, reuse = False):
    expanded_phis=[]
    for phi in phis:
        expanded_phis.append(tf.expand_dims(phi,0))
    phi = tf.concat(values=expanded_phis, axis=0)
    init_w = tf.contrib.layers.xavier_initializer()
    init_b = tf.constant_initializer(0.001)
    with tf.variable_scope('lstm', reuse = reuse):
        lstm = rnn.rnn_cell.BasicLSTMCell(n_lstm_unit, state_is_tuple=True, reuse=reuse)
        self.state_size = lstm.state_size
        step_size = tf.shape(self.x)[:1]

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c], name='c_in')
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h], name='h_in')
        state_in = [c_in, h_in]

        state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, phi, initial_state=state_in, sequence_length=time_step,
            time_major=True)

    return s


class Memory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = []
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        # transition = np.hstack((s, a, [r], s_))
        transition = [s, a, [r], s_]
        if self.pointer > self.capacity:
            index = self.pointer % self.capacity  # replace the old memory with new memory
            self.data[index] = transition
            self.pointer += 1
        else:
            self.data.append(transition)
            self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices]

class Discriminator(object):
    def __init__(self, sess, Learning_rate, S_, S, G, a):
        self.sess = sess
        self.lr = Learning_rate
        self.S = S
        self.S_ = S_
        self.Z = universe_feature_abstraction(self.S, 2, True)
        self.Z_ = universe_feature_abstraction(self.S_, 2, True)
        self.G = G
        self.a = a
        self.z_dim = self.Z.shape[1].value
        self.a_dim = self.a.shape[1].value
        with tf.variable_scope('Discriminator'):
            self.D_real, self.D_logit_real = self._build_net(scope='eval_net', s=self.Z, a=self.a, G=self.Z_,
                                                             trainable=True, reuse=False)

            self.D_fake, self.D_logit_fake = self._build_net(scope='eval_net', s=self.Z, a=self.a, G=self.G,
                                                             trainable=True, reuse=True)
            self.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Discriminator/eval_net')
            # self.D_ = self._build_net('eval_net', S_, self.a, self.G, trainable=False)
        with tf.variable_scope('D_loss'):
            self.D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logit_fake,
                                                                                      labels=tf.zeros_like(
                                                                                          self.D_logit_fake)))

            self.D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logit_real,
                                                                                      labels=tf.ones_like(
                                                                                          self.D_logit_real) * (
                                                                                                     1 - smooth)))
            self.D_loss = self.D_loss_real + self.D_loss_fake

        with tf.variable_scope('D_train'):
            t_vars = tf.trainable_variables()
            self.vars = [var for var in t_vars if var.name.startswith('Discriminator')]
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.D_loss, var_list=self.vars)

    def _build_net(self, scope, s, a, G, trainable, reuse=False):
        """
        Create the discriminator network
        param reuse: Boolean if the weights should be reused
        """
        with tf.variable_scope(scope, reuse= reuse):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.001)

            # alpha: leak relu coefficient
            alpha = 0.2
            # input layer
            with tf.variable_scope('l1'):
                n_l1 = 200
                w1_G = tf.get_variable('w1_G',[self.z_dim, n_l1], initializer=init_w, trainable= trainable)
                w1_s = tf.get_variable('w1_s',[self.z_dim, n_l1], initializer=init_w, trainable= trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.matmul(s, w1_s) + tf.matmul(G, w1_G) + tf.matmul(a, w1_a) + b1
                net = tf.maximum(alpha * net, net)
            # layer 1
            net = tf.layers.dense(net, 200,  kernel_initializer=init_w, bias_initializer= init_b,
                                  name= 'l2', trainable= trainable)
            net = tf.maximum(alpha * net, net)

            # layer 2
            net = tf.layers.dense(net, 10, kernel_initializer=init_w, bias_initializer=init_b,
                                  name='l3', trainable=trainable)
            net = tf.maximum(alpha * net, net)

            # layer 3
            with tf.variable_scope('D'):
                w3 = tf.get_variable('w3',[10, 1], initializer=init_w, trainable= trainable)
                b3 = tf.get_variable('b3', [1, 1], initializer=init_b, trainable=trainable)
                D_logit = tf.matmul(net, w3) + b3
                D = tf.nn.sigmoid(D_logit)
        return D, D_logit

    def learn(self, G_data, s_, s, a):
        self.sess.run(self.train_op, feed_dict={self.S:s, self.G:G_data, self.a:a, self.S_:s_})

    def determine(self, s, a, g):
        s = s[np.newaxis, :]
        a = a[np.newaxis, :]
        g = g[np.newaxis, :]

        return 1-self.sess.run(self.D_fake, feed_dict={self.S: s, self.G: g, self.a: a})

    def eval(self, G_data, s_, s, a):
        return self.sess.run(self.D_loss, feed_dict={self.S: s, self.G: G_data, self.a: a, self.S_: s_})


class Generator(object):
    def __init__(self, sess, action_dim,  Learning_rate, a, S):
        self.sess = sess

        self.a_dim = action_dim

        self.S = S
        self.lr = Learning_rate
        self.Z = universe_feature_abstraction(self.S, 2, True)
        self.z_dim = self.Z.shape[1].value
        self.G_dim = self.z_dim
        with tf.variable_scope('Generator'):
            self.a = a
            self.G = self._build_net(scope= 'eval_net', s=self.Z, a=self.a, trainable= True)
        t_vars = tf.trainable_variables()
        self.vars = [var for var in t_vars if var.name.startswith('Generator')]

    def _build_net(self, scope, s, a, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.001)

            # alpha: leak relu coefficient
            alpha = 0.2
            # input layer
            with tf.variable_scope('l1'):
                n_l1 = 200
                w1_s = tf.get_variable('w1_s', [self.z_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            # layer 1
            net = tf.layers.dense(net, 200, activation=tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b,
                                  name='l2', trainable=trainable)
            # layer 2
            net = tf.layers.dense(net, 50, activation=tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b,
                                  name='l3', trainable=trainable)
            # layer 3
            G = tf.layers.dense(net, self.G_dim, activation=None, kernel_initializer=init_w, bias_initializer=init_b,
                                  name='l4', trainable=trainable)
        return G

    def model_loss(self, D_logit_fake):
        with tf.variable_scope('G_loss'):
            self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake,
                                                                                 labels=tf.ones_like(
                                                                                     D_logit_fake)))
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.G_loss, var_list= self.vars)

    def learn(self, s, a):
        self.sess.run(self.train_op, feed_dict={self.S:s, self.a:a})

    def eval(self, s, a):
        return self.sess.run(self.G_loss, feed_dict={self.S: s, self.a: a})

    def predict(self, s, a):
        s = s[np.newaxis, :]
        a = a[np.newaxis, :]
        return self.sess.run(self.G, feed_dict={self.S:s, self.a:a})[0]

class Critic(object):
    def __init__(self, sess, action_dim, learning_rate, gamma, t_replace_iter, a, a_, S, S_, R, scope='Critic'):
        self.sess = sess
        self.a_dim = action_dim
        self.a = a
        self.a_ = a_
        self.S_ = S_
        self.S = S
        self.R = R
        self.lr = learning_rate
        self.gamma = gamma
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0
        self.Z = universe_feature_abstraction(self.S, 2, True)
        self.Z_ = universe_feature_abstraction(self.S_, 2, True)
        self.z_dim = self.Z.shape[1].value
        with tf.variable_scope(scope):
            # Input (s, a), output q

            self.q = self._build_net(self.Z, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(self.Z_, self.a_, 'target_net', trainable=False)    # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope+'/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope+'/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = self.R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss, var_list=self.e_params)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, a)[0]   # tensor of gradients of each sample (None, a_dim)

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.01)

            with tf.variable_scope('l1'):
                n_l1 = 200
                w1_s = tf.get_variable('w1_s', [self.z_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                  trainable=trainable)
            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
        return q

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={self.S: s, self.a: a, self.R: r, self.S_: s_})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1

class Actor(object):
    def __init__(self, sess, action_dim,  learning_rate, t_replace_iter, S, S_, scope = 'Actor', action_bound = None, time_step =4):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0
        self.S_ = S_
        self.S = S
        self.phis =[]
        self.phis.append(universe_feature_abstraction(self.S[0],2,False))
        for observation in self.S[1:]:
            self.phis.append(universe_feature_abstraction(observation, 2, True))
        self.h = build_LSTM(self.phis, 256)
        self.phi_ = universe_feature_abstraction(self.S_, 2, True)
        self.z_dim = tf.shape(self.Z)[0]
        with tf.variable_scope(scope):
            # input s, output a
            self.a = self._build_net(self.Z, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(self.Z_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope+r'/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope+'/target_net')

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.001)

            net = tf.layers.dense(s, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                  trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          name='a', trainable=trainable)
                # scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # Scale output to -action_bound to action_bound
        return actions

    def learn(self, s):   # batch update
        self.sess.run(self.train_op, feed_dict={self.S: s})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1

    def choose_action(self, s, var):
        s = s[np.newaxis, :]    # single state
        a = tf.random_normal(shape=[self.a_dim], mean=self.a, stddev=var)
        a = tf.one_hot(tf.argmax(a,1), self.a_dim)
        return self.sess.run(a, feed_dict={self.S: s})[0]  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.RMSPropOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))
    def observe(self, s):
        s = s[np.newaxis, :]
        return self.sess.run(self.Z, feed_dict = {self.S:s})[0]