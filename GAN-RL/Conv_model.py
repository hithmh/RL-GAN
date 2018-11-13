import tensorflow as tf
import numpy as np
from matplotlib import pyplot
import os
import shutil
#coding=utf-8

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
def bn_layer(x, is_training, name=None, moving_decay=0.9, eps=1e-5, reuse=False):

    shape = x.shape
    assert len(shape) in [2,4]
    init_w = tf.contrib.layers.xavier_initializer()
    init_b = tf.constant_initializer(0.001)
    param_shape = shape[-1]
    with tf.variable_scope(name+'_BatchNorm', reuse=reuse):

        gamma = tf.get_variable('gamma',param_shape,initializer=init_w)
        beta = tf.get_variable('beat', param_shape,initializer=init_b)


        axes = list(range(len(shape)-1))
        batch_mean, batch_var = tf.nn.moments(x,axes,name='moments')


        ema = tf.train.ExponentialMovingAverage(moving_decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean,batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        # with tf.variable_scope('', reuse=tf.AUTO_REUSE):
        mean, var = tf.cond(tf.equal(is_training, True), mean_var_with_update,lambda: (ema.average(batch_mean), ema.average(batch_var)))

        return tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)


def universe_feature_abstraction(s, nConvs=4, reuse=False, time_step=4, output_dim=32, norm = True, is_training=False):
    print('Using universe head design')
    s= tf.reshape(s,shape=[-1]+s.get_shape().as_list()[2:])
    init_w = tf.contrib.layers.xavier_initializer()
    init_b = tf.constant_initializer(0.001)
    with tf.variable_scope('feature_abstraction', reuse = reuse):
        for i in range(nConvs):
            if norm:
                s = tf.layers.conv2d(s, 32, [3, 3], [2, 2], name="l{}".format(i + 1), activation=None,
                                    )
                s = tf.layers.batch_normalization(s, training=is_training, name="bn{}".format(i + 1))
                s = tf.nn.elu(s)
            else:
                s = tf.layers.conv2d(s, 32, [3, 3], [2, 2], name="l{}".format(i + 1), activation=tf.nn.elu,
                                     )

            # print('Loop{} '.format(i+1),tf.shape(x))
            # print('Loop{}'.format(i+1),x.get_shape())
        s = tf.reshape(s, [-1, time_step, np.prod(s.get_shape().as_list()[1:])])

        #fc_layer 1
        # s = tf.layers.dense(s, 256, name='dense1', kernel_initializer=init_w, bias_initializer=init_b,
        #                     activation=None)
        # if norm:
        #     s = tf.layers.batch_normalization(s, training=is_training, name="bn{}".format(nConvs + 1))
        # s = tf.nn.elu(s)
        #
        # # fc_layer 2
        # s = tf.layers.dense(s, output_dim, name='dense2', kernel_initializer=init_w, bias_initializer=init_b,
        #                     activation=None)
        # if norm:
        #     s = tf.layers.batch_normalization(s, training=is_training, name="bn{}".format(nConvs + 2))
        # s = tf.nn.elu(s)
    return s


class lstm_unit(object):

    def __init__(self, phis, phis_, s, sess, phi_is_training, n_lstm_unit=256, time_step=4):
        self.c_in = tf.placeholder(tf.float32, [None, n_lstm_unit], name='c_in')
        self.h_in = tf.placeholder(tf.float32, [None, n_lstm_unit], name='h_in')
        self.c_in_ = tf.placeholder(tf.float32, [None, n_lstm_unit], name='c_in')
        self.h_in_ = tf.placeholder(tf.float32, [None, n_lstm_unit], name='h_in')
        self.lstm_state, self.lstm = self.build_LSTM(phis, self.c_in, self.h_in, n_lstm_unit, time_step,reuse = False)
        # self.lstm_state_test, self.lstm_test = self.build_LSTM(phis_inf, self.c_in, self.h_in, n_lstm_unit, time_step, reuse=True)
        if phis_ is not None:
            self.lstm_state_, _ = self.build_LSTM(phis_,self.c_in_,self.h_in_, n_lstm_unit, time_step, reuse=True)
        self.S = s
        self.sess = sess
        c_init = np.zeros((1, self.lstm.state_size.c), np.float32)
        h_init = np.zeros((1, self.lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        self.phi_is_training = phi_is_training

    def build_LSTM(self, phis, c_in, h_in, n_lstm_unit = 256, time_step =4, reuse = False):
        with tf.variable_scope('lstm', reuse=reuse):
            # phis = tf.layers.dense(phis, n_lstm_unit, name='input',
            # kernel_initializer=init_w, bias_initializer=init_b,activation=tf.nn.elu)
            lstm = tf.contrib.rnn.BasicLSTMCell(n_lstm_unit, state_is_tuple=True, reuse=reuse)
            # state_init = lstm.zero_state(batch_size, dtype=tf.float32)
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm, phis, initial_state=state_in,
                time_major=False)
        return lstm_state, lstm

    def get_state(self, s, lstm_state):
        s = np.expand_dims(s,0)
        return self.sess.run(self.lstm_state, feed_dict={self.S: s, self.c_in: lstm_state[0], self.h_in: lstm_state[1],
                                                         self.phi_is_training:False})

    def get_initial_state(self):
        return self.state_init


class Memory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.data_s = []
        self.data_g = []
        self.data_D_lstm_s = []
        self.data_G_lstm_s = []
        self.data_a = []
        self.data_r = []
        self.data_curious_r = []
        self.data_s_ = []
        self.data_D_lstm_s_ = []
        self.data_G_lstm_s_ = []
        self.pointer = 0

    def store_transition(self, s, g, D_lstm_s, G_lstm_s, a, r, curious_r, s_, D_lstm_s_, G_lstm_s_):
        # transition = np.hstack((s, a, [r], s_))

        if self.pointer > self.capacity:
            index = self.pointer % self.capacity  # replace the old memory with new memory
            self.data_s[index] = s
            self.data_g[index] = g
            self.data_D_lstm_s[index] = D_lstm_s
            self.data_G_lstm_s[index] = G_lstm_s
            self.data_a[index] = a
            self.data_r[index] = r
            self.data_curious_r[index] = curious_r
            self.data_s_[index] = s_
            self.data_D_lstm_s_[index] = D_lstm_s_
            self.data_G_lstm_s_[index] = G_lstm_s_
            self.pointer += 1
        else:
            self.data_s.append(s)
            self.data_g.append(g)
            self.data_a.append(a)
            self.data_r.append(r)
            self.data_curious_r.append(curious_r)
            self.data_s_.append(s_)
            self.data_D_lstm_s.append(D_lstm_s)
            self.data_D_lstm_s_.append(D_lstm_s_)
            self.data_G_lstm_s.append(G_lstm_s)
            self.data_G_lstm_s_.append(G_lstm_s_)
            self.pointer += 1
            self.len = self.pointer

    def sample(self, n):
        # assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.len, size=n)
        b_s = np.array(self.data_s)[indices]
        b_g = np.array(self.data_g)[indices]
        b_a = np.array(self.data_a)[indices]
        b_r = np.array(self.data_r)[indices]
        b_curious_r = np.array(self.data_curious_r)[indices]
        b_s_ = np.array(self.data_s_)[indices]
        b_D_lstm_s = np.squeeze(np.array(self.data_D_lstm_s)[indices], 2)
        b_D_lstm_s_ = np.squeeze(np.array(self.data_D_lstm_s_)[indices],2)
        b_G_lstm_s = np.squeeze(np.array(self.data_G_lstm_s)[indices], 2)
        b_G_lstm_s_ = np.squeeze(np.array(self.data_G_lstm_s_)[indices], 2)
        return b_s,b_g,b_D_lstm_s, b_G_lstm_s, b_a, b_r, b_curious_r, b_s_, b_D_lstm_s_, b_G_lstm_s_

class Discriminator(object):
    def __init__(self, sess, learning_rate, single_s_, s, g_h, g, a, lstm_unit, g_lstm_unit, observation, phi_is_training,
                 g_is_training, a_is_training, d_is_training, batch_size, l2_regularizer_weight, mode):
        self.sess = sess
        self.lr = learning_rate
        self.l2_weight = l2_regularizer_weight
        self.batch_size = batch_size
        self.S = s
        self.d_is_training = d_is_training
        self.phi_is_training = phi_is_training
        self.single_S_ = single_s_
        self.h = lstm_unit.lstm_state[1]
        self.h_in = lstm_unit.h_in
        self.c_in = lstm_unit.c_in
        self.g_h_in = g_lstm_unit.h_in
        self.g_c_in = g_lstm_unit.c_in
        self.lstm_state_size = lstm_unit.c_in.shape
        self.G_h = g_h
        self.G = g
        self.g_is_training = g_is_training
        self.obs = observation
        self.a = a
        self.a_is_training = a_is_training
        self.h_dim = self.h.shape[1].value
        self.obs_dim = self.obs.shape[1].value
        self.a_dim = self.a.shape[1].value
        self.mode = mode
        with tf.variable_scope('Discriminator'):
            self.D_real, self.D_logit_real, self.unscaled_D_real = self._build_net(scope='eval_net', s=self.h,
                                                                                   a=self.a, G=self.obs,
                                                                                   trainable=True, reuse=False,
                                                                                   d_is_training=self.d_is_training)
            if mode=='full_prediction':
                self.D_fake, self.D_logit_fake, self.unscaled_D_fake = self._build_net(scope='eval_net', s=self.h,
                                                                                       a=self.a, G=self.G_h,
                                                                                       trainable=True, reuse=True,
                                                                                       d_is_training=False)
            else:
                self.D_fake, self.D_logit_fake, self.unscaled_D_fake = self._build_net(scope='eval_net', s=self.h,
                                                                                       a=self.a, G=self.G,
                                                                                       trainable=True, reuse=True,
                                                                                       d_is_training=False)

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
            t_vars = tf.trainable_variables()
            self.vars = [var for var in t_vars if var.name.startswith('Discriminator')
                         or var.name.startswith('LSTM_feature_abstraction')]
            for var in self.vars:
                if var.name.endswith('kernel:0'):
                    tf.add_to_collection('D_losses',tf.contrib.layers.l2_regularizer(self.l2_weight)(var))
            self.D_loss = self.D_loss_real + self.D_loss_fake+ tf.add_n(tf.get_collection('D_losses'))

        with tf.variable_scope('D_train'):

            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.D_loss, var_list=self.vars)

    def _build_net(self, scope, s, a, G, d_is_training, trainable=True, reuse=False, norm=True):
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
                w1_G = tf.get_variable('w1_G',[self.obs_dim, n_l1], initializer=init_w, trainable= trainable)

                w1_s = tf.get_variable('w1_s',[self.h_dim, n_l1], initializer=init_w, trainable= trainable)

                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                if not reuse:
                    tf.add_to_collection('D_losses', tf.contrib.layers.l2_regularizer(self.l2_weight)(w1_G))
                    tf.add_to_collection('D_losses', tf.contrib.layers.l2_regularizer(self.l2_weight)(w1_s))
                    tf.add_to_collection('D_losses', tf.contrib.layers.l2_regularizer(self.l2_weight)(w1_a))
                net = tf.matmul(s, w1_s) + tf.matmul(G, w1_G) + tf.matmul(a, w1_a) + b1
                if norm:
                    net = tf.layers.batch_normalization(net, training=d_is_training)
                net = tf.maximum(alpha * net, net)
            # layer 1
            net = tf.layers.dense(net, 200,  kernel_initializer=init_w, bias_initializer= init_b,
                                  name= 'l2', trainable= trainable)
            if norm:
                net = tf.layers.batch_normalization(net, training=d_is_training)
            net = tf.maximum(alpha * net, net)

            # layer 2
            net = tf.layers.dense(net, 10, kernel_initializer=init_w, bias_initializer=init_b,
                                  name='l3', trainable=trainable)
            if norm:
                net = tf.layers.batch_normalization(net, training=d_is_training)
            D_logit = tf.maximum(alpha * net, net)

            # layer 3
            with tf.variable_scope('D'):

                unscaled_D = tf.layers.dense(D_logit, 1,  kernel_initializer=init_w, bias_initializer= init_b,
                                  name= 'l2', trainable= trainable)
                D = tf.nn.sigmoid(unscaled_D)
        return D, D_logit,unscaled_D

    def learn(self, G_data, s_, s, b_lstm_s, a, b_g_lstm_s):
        s_ = np.expand_dims(s_[:,-1,:],1)
        feed_dict = {self.S: s, self.a: a, self.single_S_: s_,
                     self.h_in: b_lstm_s[:, 1, :], self.c_in: b_lstm_s[:, 0, :],
                     self.g_h_in: b_g_lstm_s[:, 1, :], self.g_c_in: b_g_lstm_s[:, 0, :],
                     self.d_is_training: True, self.phi_is_training: True,
                     self.a_is_training: False, self.g_is_training: False}
        if self.mode=='full_prediction':
            feed_dict[self.G] = G_data
        else:
            feed_dict[self.G_h] = G_data
        self.sess.run(self.train_op, feed_dict=feed_dict)

    def determine(self, s, lstm_s, a, g):
        s = s[np.newaxis, :]
        a = a[np.newaxis, :]
        g = g[np.newaxis, :]
        feed_dict = {self.S: s, self.a: a, self.h_in: lstm_s[1],
                     self.c_in: lstm_s[0], self.d_is_training: False,
                     self.phi_is_training: False, self.a_is_training: False,
                     self.g_is_training: False}
        if self.mode=='full_prediction':
            feed_dict[self.G] = g
        else:
            feed_dict[self.G_h] = g
        return 0.5-self.sess.run(self.D_fake,feed_dict=feed_dict)

    def determine_batch(self, b_s, b_lstm_s, b_a, b_g):

        return np.ones([b_s.shape[0],1])-self.sess.run(self.D_fake, feed_dict={self.S: b_s, self.G: b_g, self.a: b_a,
                                                                               self.h_in: b_lstm_s[:, 1, :],
                                                                               self.c_in: b_lstm_s[:, 0, :],
                                                                               self.d_is_training: False,
                                                                               self.phi_is_training: False,
                                                                               self.a_is_training: False,
                                                                               self.g_is_training: False
                                                                               })

    def observe_and_compare(self, s_, g):
        single_s_ = s_[-1]
        single_s_ = np.expand_dims(single_s_, axis=0)
        single_s_ = np.expand_dims(single_s_, axis=0)
        if self.mode =='hidden_state':
            obs_ = self.sess.run(self.obs, feed_dict={self.single_S_: single_s_, self.phi_is_training: False})[0]
            return np.sum(np.square(obs_-g))
        else:
            return 0.


    def eval(self, b_g, b_s_, b_s, b_lstm_s, b_a, b_g_lstm_s):

        b_single_s_ = np.expand_dims(b_s_[:, -1, :], 1)
        feed_dict = {self.S: b_s, self.a: b_a, self.single_S_: b_single_s_,
                     self.h_in: b_lstm_s[:, 1, :], self.c_in: b_lstm_s[:, 0, :],
                     self.g_h_in: b_g_lstm_s[:, 1, :], self.g_c_in: b_g_lstm_s[:, 0, :],
                     self.d_is_training: False, self.phi_is_training: False,
                     self.a_is_training: False, self.g_is_training: False}
        if self.mode== 'full_prediction':
            feed_dict[self.G] = b_g
        else:
            feed_dict[self.G_h] = b_g

        return self.sess.run(self.D_loss, feed_dict=feed_dict)


class Generator(object):
    def __init__(self, sess, action_dim,  Learning_rate, batch_size, a, S, LSTM_unit, d_LSTM_unit, phi_dim, batch_a,
                 l2_regularizer_weight, phi_is_training, phi_gen_is_training, g_is_training, a_is_training, d_is_training,
                 mode):
        self.sess = sess
        self.batch_size = batch_size
        self.l2_weight = l2_regularizer_weight
        self.a_dim = action_dim
        self.a = a
        self.batch_a = batch_a
        self.S = S
        self.lr = Learning_rate
        self.h = LSTM_unit.lstm_state[1]
        self.h_in = LSTM_unit.h_in
        self.c_in = LSTM_unit.c_in
        self.d_h_in = d_LSTM_unit.h_in
        self.d_c_in = d_LSTM_unit.c_in
        self.lstm_state_size = LSTM_unit.c_in.shape
        self.h_dim = self.h.shape[1].value
        self.G_dim = phi_dim
        self.phi_is_training = phi_is_training
        self.g_is_training = g_is_training
        self.a_is_training = a_is_training
        self.d_is_training = d_is_training
        self.phi_gen_is_training = phi_gen_is_training
        self.mode = mode
        with tf.variable_scope('Generator'):
            if mode =='hidden_state':
                self.G = self._build_net('eval_net', self.h, self.a, True, False, self.g_is_training)
                self.G_batch = self._build_net('eval_net', self.h, self.batch_a, False, True, False)
            else:
                self.G = self._build_full_generator('eval_net', self.h, self.a, True, False, self.g_is_training)
                self.G_batch = self._build_full_generator('eval_net', self.h, self.batch_a, False, True, False)

        t_vars = tf.trainable_variables()
        self.vars = [var for var in t_vars if var.name.startswith('Generator')or var.name.startswith('G_')]
        for var in self.vars:
            if var.name.endswith('kernel:0'):
                tf.add_to_collection('G_losses', tf.contrib.layers.l2_regularizer(self.l2_weight)(var))

    def _build_net(self, scope, s, a, trainable, reuse, g_is_training, norm=True):
        with tf.variable_scope(scope, reuse=reuse):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.001)

            # alpha: leak relu coefficient
            alpha = 0.2
            # input layer
            with tf.variable_scope('l1'):
                n_l1 = 200
                w1_s = tf.get_variable('w1_s', [self.h_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                if not reuse:
                    tf.add_to_collection('G_losses', tf.contrib.layers.l2_regularizer(self.l2_weight)(w1_s))
                    tf.add_to_collection('G_losses', tf.contrib.layers.l2_regularizer(self.l2_weight)(w1_a))
                net = tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1
                if norm:
                    net = tf.layers.batch_normalization(net,training=g_is_training)
                net = tf.nn.elu(net)
            # layer 1
            net = tf.layers.dense(net, 50, kernel_initializer=init_w, bias_initializer=init_b,
                                  name='l2', trainable=trainable)
            if norm:
                net = tf.layers.batch_normalization(net, training=g_is_training)
            net = tf.nn.elu(net)
            # layer 2
            net = tf.layers.dense(net, 50, kernel_initializer=init_w, bias_initializer=init_b,
                                  name='l3', trainable=trainable)
            if norm:
                net = tf.layers.batch_normalization(net, training=g_is_training)
            net = tf.nn.elu(net)
            # layer 2
            net = tf.layers.dense(net, 50, kernel_initializer=init_w, bias_initializer=init_b,
                                  name='l3-2', trainable=trainable)
            if norm:
                net = tf.layers.batch_normalization(net, training=g_is_training)
            net = tf.nn.elu(net)
            # layer 3
            G = tf.layers.dense(net, self.G_dim, activation=None, kernel_initializer=init_w, bias_initializer=init_b,
                                  name='l4', trainable=trainable)
            # if norm:
            #     G = tf.layers.batch_normalization(G, training=g_is_training)
        return G

    def _build_full_generator(self, scope, s, a, trainable, reuse, g_is_training, norm=True):
        with tf.variable_scope(scope, reuse=reuse):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.001)
            # alpha: leak relu coefficient
            alpha = 0.2
            # input layer
            with tf.variable_scope('l1'):
                n_l1 = 200
                w1_s = tf.get_variable('w1_s', [self.h_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                if not reuse:
                    tf.add_to_collection('G_losses', tf.contrib.layers.l2_regularizer(self.l2_weight)(w1_s))
                    tf.add_to_collection('G_losses', tf.contrib.layers.l2_regularizer(self.l2_weight)(w1_a))
                net = tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1
                if norm:
                    net = tf.layers.batch_normalization(net,training=g_is_training)
                net = tf.nn.elu(net)
            # layer 1
            net = tf.layers.dense(net, 500, kernel_initializer=init_w, bias_initializer=init_b,
                                  name='l2', trainable=trainable)
            if norm:
                net = tf.layers.batch_normalization(net, training=g_is_training)
            net = tf.nn.elu(net)
            # layer 2
            net = tf.layers.dense(net, 1000, kernel_initializer=init_w, bias_initializer=init_b,
                                  name='l3', trainable=trainable)
            if norm:
                net = tf.layers.batch_normalization(net, training=g_is_training)
            net = tf.nn.elu(net)
            # layer 2
            net = tf.layers.dense(net, 1500, kernel_initializer=init_w, bias_initializer=init_b,
                                  name='l3-2', trainable=trainable)
            if norm:
                net = tf.layers.batch_normalization(net, training=g_is_training)
            net = tf.nn.elu(net)
            # layer 3
            G = tf.layers.dense(net, 1764, activation=None, kernel_initializer=init_w, bias_initializer=init_b,
                                  name='l4', trainable=trainable)
            # if norm:
            #     G = tf.layers.batch_normalization(G, training=g_is_training)
            G = tf.reshape(G, [-1,1,42,42,1])
        return G

    def model_loss(self, D_logit_fake):
        with tf.variable_scope('G_loss'):
            G_mseloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake,
                                                                                 labels=tf.ones_like(
                                                                                     D_logit_fake)))
            normalization_loss = tf.add_n(tf.get_collection('G_losses'))
            self.G_loss = G_mseloss + normalization_loss
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.G_loss, var_list= self.vars)

    def learn(self, s,b_lstm_s, a, b_d_lstm_s):
        self.sess.run(self.train_op, feed_dict={self.S: s, self.a: a, self.h_in: b_lstm_s[:, 1, :],
                                                self.c_in: b_lstm_s[:, 0, :],
                                                self.d_h_in: b_d_lstm_s[:, 1, :],
                                                self.d_c_in: b_d_lstm_s[:, 0, :],
                                                self.phi_is_training: False,
                                                self.g_is_training: True,
                                                self.a_is_training: False,
                                                self.phi_gen_is_training: True,
                                                self.d_is_training: False})

    def eval(self, s, b_lstm_s, a, b_d_lstm_s):

        return self.sess.run(self.G_loss, feed_dict={self.S: s, self.a: a,
                                                     self.h_in: b_lstm_s[:, 1, :],
                                                     self.c_in: b_lstm_s[:, 0, :],
                                                     self.d_h_in: b_d_lstm_s[:, 1, :],
                                                     self.d_c_in: b_d_lstm_s[:, 0, :],
                                                     self.phi_is_training: False,
                                                     self.g_is_training: False,
                                                     self.a_is_training: False,
                                                     self.d_is_training: False,
                                                     self.phi_gen_is_training: False
                                                     })

    def predict(self, s,lstm_state, a):
        s = s[np.newaxis, :]
        a = a[np.newaxis, :]
        return self.sess.run(self.G, feed_dict={self.S: s, self.a: a,
                                                self.h_in: lstm_state[1],
                                                self.c_in: lstm_state[0],
                                                self.phi_is_training: False,
                                                self.g_is_training: False,
                                                self.a_is_training: False,
                                                self.phi_gen_is_training: False
                                                })[0]

    def predict_batch(self, b_s, b_lstm_s, b_a):
        return self.sess.run(self.G_batch, feed_dict={self.S:b_s, self.batch_a: b_a,
                                                      self.h_in: b_lstm_s[:, 1, :],
                                                      self.c_in: b_lstm_s[:, 0, :],
                                                      self.phi_is_training: False,
                                                      self.g_is_training: False,
                                                      self.a_is_training: False,
                                                      self.phi_gen_is_training: False
                                                      })

class Critic(object):
    def __init__(self, sess, action_dim, learning_rate, batch_size, gamma, t_replace_iter, a, a_, LSTM_unit, S, S_, R,
                 phi_is_training, c_is_training, a_is_training, norm, scope='Critic'):
        self.sess = sess
        self.a_dim = action_dim
        self.a = a
        self.a_ = a_
        self.S_ = S_
        self.S = S
        self.h = LSTM_unit.lstm_state[1]
        self.h_ = LSTM_unit.lstm_state_[1]
        self.h_in = LSTM_unit.h_in
        self.c_in = LSTM_unit.c_in
        self.h_in_ = LSTM_unit.h_in_
        self.c_in_ = LSTM_unit.c_in_
        self.lstm_state_size = LSTM_unit.c_in.shape
        self.R = R
        self.lr = learning_rate
        self.batch_size = batch_size
        self.gamma = gamma
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        self.h_dim = self.h.shape[1].value
        self.phi_is_training = phi_is_training
        self.c_is_training = c_is_training
        self.a_is_training = a_is_training
        with tf.variable_scope(scope):
            # Input (s, a), output q

            self.q = self._build_net(self.h, self.a, 'eval_net', False, self.c_is_training, norm)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(self.h_, self.a_, 'target_net', False, False, norm)    # target_q is based on a_ from Actor's target_net

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

    def _build_net(self, s, a, scope, trainable, c_is_training, norm=True):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.01)

            with tf.variable_scope('l1'):
                n_l1 = 200
                w1_s = tf.get_variable('w1_s', [self.h_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1
                if norm:
                    net = tf.layers.batch_normalization(net,training=c_is_training)
                net = tf.nn.relu6(net)

            net = tf.layers.dense(net, 200,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            if norm:
                net = tf.layers.batch_normalization(net, training=c_is_training)
            net = tf.nn.relu6(net)

            net = tf.layers.dense(net, 10,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                  trainable=trainable)
            if norm:
                net = tf.layers.batch_normalization(net, training=c_is_training)
            net = tf.nn.relu6(net)
            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
        return q

    def learn(self, s, b_lstm_s, a, r, s_, b_lstm_s_):
        self.sess.run(self.train_op, feed_dict={self.S: s, self.a: a, self.R: r, self.S_: s_,
                      self.h_in: b_lstm_s[:, 1, :], self.c_in: b_lstm_s[:, 0, :],
                      self.h_in_: b_lstm_s_[:, 1, :], self.c_in_: b_lstm_s_[:, 0, :],
                                                self.phi_is_training: False,
                                                self.c_is_training: True,
                                                self.a_is_training: False
                                                })
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1

class Actor(object):
    def __init__(self, sess, action_dim,  learning_rate,batch_size, t_replace_iter, LSTM_unit, S, S_,
                 phi_is_training, a_is_training,c_is_training, norm, scope = 'Actor', action_bound = None):
        self.sess = sess
        self.batch_size = batch_size
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0
        self.h = LSTM_unit.lstm_state[1]
        self.h_ = LSTM_unit.lstm_state_[1]
        self.h_in = LSTM_unit.h_in
        self.c_in = LSTM_unit.c_in
        self.h_in_ = LSTM_unit.h_in_
        self.c_in_ = LSTM_unit.c_in_
        self.lstm_state_size = LSTM_unit.c_in.shape
        self.S = S
        self.S_ = S_
        self.phi_is_training = phi_is_training
        self.a_is_training = a_is_training
        self.c_is_training = c_is_training

        with tf.variable_scope(scope):
            # input s, output a
            self.a = self._build_net(self.h, scope='eval_net', trainable=True, norm=norm, a_is_training=self.a_is_training)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(self.h_, scope='target_net', trainable=False, a_is_training=False, norm=norm)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope+r'/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope+'/target_net')

    def _build_net(self, h, scope, trainable, norm, a_is_training):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.001)

            net = tf.layers.dense(h, 200,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            if norm:
                net = tf.layers.batch_normalization(net, training=a_is_training)
            net = tf.nn.relu6(net)

            net = tf.layers.dense(net, 200,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            if norm:
                net = tf.layers.batch_normalization(net, training=a_is_training)
            net = tf.nn.relu6(net)

            net = tf.layers.dense(net, 10,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                  trainable=trainable)
            if norm:
                net = tf.layers.batch_normalization(net, training=a_is_training)
            net = tf.nn.relu6(net)

            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, kernel_initializer=init_w,
                                          name='a', trainable=trainable)
                if norm:
                    actions = tf.layers.batch_normalization(actions, training=a_is_training)
                actions = tf.nn.tanh(actions)
                # scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # Scale output to -action_bound to action_bound
        return actions

    def learn(self, s, b_lstm_s):   # batch update
        self.sess.run(self.train_op, feed_dict={self.S: s, self.h_in:b_lstm_s[:, 1, :],self.c_in: b_lstm_s[:, 0, :],
                                                self.phi_is_training: False,
                                                self.a_is_training: True,
                                                self.c_is_training: False,
                                                })
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1

    def choose_action(self, s, lstm_state, var):
        s = s[np.newaxis, :]    # single state
        a = self.sess.run(self.a, feed_dict={self.S: s,self.h_in: lstm_state[1], self.c_in: lstm_state[0],
                                             self.phi_is_training: False,
                                             self.a_is_training: False,
                                             })
        a = np.random.normal(a, var)
        a_onehot = np.zeros(self.a_dim, dtype=np.int)
        a_onehot[a.argmax(axis=1)] = 1
        return a_onehot  # single action

    def choose_mario_action(self,s, lstm_state, var):
        s = s[np.newaxis, :]    # single state
        a = self.sess.run(self.a, feed_dict={self.S: s,self.h_in: lstm_state[1], self.c_in: lstm_state[0],
                                             self.phi_is_training: False,
                                             self.a_is_training: False,
                                             })
        a = np.random.normal(a, var)
        a_onehot = np.zeros(self.a_dim, dtype=np.int)
        a_onehot[a.argmax(axis=1)] = 1
        return a_onehot  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.RMSPropOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))