import tensorflow as tf
import numpy as np
import shutil
import os
from arm_env import ArmEnv
from DDPG import Actor, Critic
import matplotlib.pyplot as plt



class Discriminator(object):
    def __init__(self, sess, state_dim, action_dim,  Learning_rate, a, G, S_, S):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.G_dim = state_dim
        self.lr = Learning_rate

        with tf.variable_scope('Discriminator'):
            self.a = a
            self.G = G
            self.S = S
            self.S_ = S_
            self.D_fake, self.D_logit_fake = self._build_net(scope='eval_net', s=self.S, a=self.a, G=self.G, trainable=True, reuse=False)
            self.D_real, self.D_logit_real = self._build_net(scope='eval_net', s=self.S, a=self.a, G=self.S_, trainable=True, reuse=True)
            self.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Discriminator/eval_net')


            # self.D_ = self._build_net('eval_net', S_, self.a, self.G, trainable=False)

        with tf.variable_scope('D_loss'):

            self.D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logit_fake,
                                                                                 labels=tf.zeros_like(
                                                                                     self.D_logit_fake)))

            self.D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logit_real,
                                                                                 labels=tf.ones_like(
                                                                                     self.D_logit_real)*(1-smooth)))
            self.D_loss = self.D_loss_real + self.D_loss_fake

        with tf.variable_scope('D_train'):
            t_vars = tf.trainable_variables()
            self.vars = [var for var in t_vars if var.name.startswith('Discriminator')]
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.D_loss, var_list= self.vars)
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
                w1_G = tf.get_variable('w1_G',[self.G_dim, n_l1], initializer=init_w, trainable= trainable)
                w1_s = tf.get_variable('w1_s',[self.s_dim, n_l1], initializer=init_w, trainable= trainable)
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
    def __init__(self, sess, state_dim, action_dim,  Learning_rate, a, S):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.G_dim = state_dim
        self.lr = Learning_rate

        with tf.variable_scope('Generator'):
            self.a = a
            self.S = S
            self.G = self._build_net(scope= 'eval_net', s=self.S, a=self.a, trainable= True)
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
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
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

class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s,  s_, G, a, r, curious_r):
        transition = np.hstack((s,  s_, G, a, [r], curious_r))
        index = self.pointer % self.capacity
        self.data[index, :] = transition
        self.pointer += 1
    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]


def train():
    var = 2.
    pointer = 0


    J_D_loss = np.zeros((N_trials, len(N_vals)))
    J_G_loss = np.zeros((N_trials, len(N_vals)))
    for ep in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        ep_curious_reward = 0
        D_loss = 0
        G_loss = 0
        l_2_loss = 0
        for t in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            a = actor.choose_action(s)
            a = np.clip(np.random.normal(a, var), *ACTION_BOUND)
            g = generator.predict(s, a)
            s_, r, done = env.step(a)
            curious_r = r + ITA * discriminator.determine(s,a,g)[0]
            M.store_transition(s,s_,g,a,r,curious_r)
            l_2_loss += np.sum(np.square(s_-g))
            if M.pointer > MEMORY_CAPACITY:
                var = max([var*.9999,VAR_MIN])
                b_M = M.sample(BATCH_SIZE)
                b_s = b_M[:, :STATE_DIM]
                b_s_ = b_M[:, STATE_DIM:2*STATE_DIM]
                b_g = b_M[:, STATE_DIM*2:STATE_DIM*3]
                b_a = b_M[:, -2-ACTION_DIM:-2]
                b_r = b_M[:, -2:-1]
                b_curious_r = b_M[:, -1:]

                # Learn the minibatch
                critic.learn(b_s, b_a, b_curious_r, b_s_)
                actor.learn(b_s)
                discriminator.learn(b_g, b_s_, b_s, b_a)
                generator.learn(b_s,b_a)
                D_loss += discriminator.eval(b_g, b_s_, b_s, b_a)
                G_loss += generator.eval(b_s,b_a)




            s = s_
            ep_reward += r
            ep_curious_reward +=curious_r
    
            if t == MAX_EP_STEPS - 1 or done:
                # if done:
                result = '| done' if done else '| ----'
                print('Ep:', ep,
                      result,
                      '| R: %i' % int(ep_reward),
                      '| Curious_R: %f' % float(ep_curious_reward),
                      '| D_loss: %f' % float(D_loss/t),
                      '| G_loss: %f' % float(G_loss/t),
                      '| Prediction_error: %f' %float(l_2_loss/t),
                      '| Explore: %.2f' % var,
                      )
                break

        if ep == N_vals[pointer]:

            # evaluate the minibatch
            J_D_loss[0, pointer] = D_loss/t
            J_G_loss[0, pointer] = G_loss/t
            J_Curious[0,pointer] = ep_curious_reward
            J_r[0, pointer] = ep_reward
            if pointer<N_vals.__len__()-1:
                pointer += 1

                
    if os.path.isdir(path): shutil.rmtree(path)
    os.mkdir(path)
    ckpt_path = os.path.join('./'+MODE[n_model], 'Curious_GAN.ckpt')
    save_path = saver.save(sess, ckpt_path, write_meta_graph=False)
    print("\nSave Model %s\n" % save_path)


def train_DDPG():
    var = 2.
    pointer = 0

    for ep in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        for t in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            a = actor_DDPG.choose_action(s)
            a = np.clip(np.random.normal(a, var), *ACTION_BOUND)
            s_, r, done = env.step(a)
            M_DDPG.store_transition(s, s_, s_, a, r, r)
            if M_DDPG.pointer > MEMORY_CAPACITY:
                var = max([var * .9999, VAR_MIN])
                b_M = M_DDPG.sample(BATCH_SIZE)
                b_s = b_M[:, :STATE_DIM]
                b_s_ = b_M[:, STATE_DIM:2*STATE_DIM]

                b_a = b_M[:, -2-ACTION_DIM:-2]
                b_r = b_M[:, -2:-1]


                # Learn the minibatch
                critic_DDPG.learn(b_s, b_a, b_r, b_s_)
                actor_DDPG.learn(b_s)

            s = s_
            ep_reward += r

            if t == MAX_EP_STEPS - 1 or done:
                # if done:
                result = '| done' if done else '| ----'
                print('Ep:', ep,
                      result,
                      '| R: %i' % int(ep_reward),
                      '| Explore: %.2f' % var,
                      )
                break

        if ep == N_vals[pointer]:
            # evaluate the minibatch

            J_r_DDPG[0, pointer] = ep_reward
            if pointer < N_vals.__len__()-1:
                pointer += 1

    if os.path.isdir(path): shutil.rmtree(path)
    os.mkdir(path)
    ckpt_path = os.path.join('./' + MODE[n_model], 'DDPG.ckpt')
    save_path = saver.save(sess, ckpt_path, write_meta_graph=False)
    print("\nSave Model %s\n" % save_path)
def eval():
    env.set_fps(30)
    s = env.reset()
    while True:
        if RENDER:
            env.render()
        a = actor.choose_action(s)
        s_, r, done = env.step(a)
        s = s_

if __name__ == '__main__':
    smooth = 0
    np.random.seed(1)
    tf.set_random_seed(1)

    MAX_EPISODES = 600
    MAX_EP_STEPS = 200
    LR_D = 1e-4  # learning rate for actor
    LR_G = 1e-4  # learning rate for critic
    LR_A = 1e-4  # learning rate for actor
    LR_C = 1e-4  # learning rate for critic
    GAMMA = 0.9  # reward discount
    REPLACE_ITER_A = 1100
    REPLACE_ITER_C = 1000
    MEMORY_CAPACITY = 5000
    BATCH_SIZE = 16
    VAR_MIN = 0.1
    RENDER = True
    LOAD = False
    MODE = ['easy', 'hard']
    n_model = 1
    ITA = 0.01  # Curious coefficient

    env = ArmEnv(mode=MODE[n_model])
    STATE_DIM = env.state_dim
    ACTION_DIM = env.action_dim
    ACTION_BOUND = env.action_bound

    with tf.name_scope('S'):
        S = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s')
    with tf.name_scope('R'):
        R = tf.placeholder(tf.float32, [None, 1], name='r')
    with tf.name_scope('S_'):
        S_ = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s_')
    with tf.name_scope('G'):
        G = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='G')
    sess = tf.Session()
    actor = Actor(sess, ACTION_DIM, ACTION_BOUND[1], LR_A, REPLACE_ITER_A, S=S, S_=S_)
    critic = Critic(sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, REPLACE_ITER_C, actor.a, actor.a_, S=S, S_=S_, R=R)
    actor.add_grad_to_graph(critic.a_grads)

    generator = Generator(Learning_rate=LR_G, a=actor.a, S=S, action_dim=ACTION_DIM, sess=sess, state_dim=STATE_DIM)
    discriminator = Discriminator(G=generator.G, Learning_rate=LR_D, S=S, S_=S_, a=actor.a, action_dim=ACTION_DIM,
                                  sess=sess, state_dim=STATE_DIM)
    generator.model_loss(discriminator.D_logit_fake)
    M = Memory(MEMORY_CAPACITY, dims=3 * STATE_DIM + ACTION_DIM + 2)

    actor_DDPG = Actor(sess, ACTION_DIM, ACTION_BOUND[1], LR_A, REPLACE_ITER_A, S=S, S_=S_, scope='actor_DDPG')
    critic_DDPG = Critic(sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, REPLACE_ITER_C, actor_DDPG.a, actor_DDPG.a_, S=S,
                         S_=S_, R=R, scope='critic_DDPG')
    actor_DDPG.add_grad_to_graph(critic_DDPG.a_grads)

    M_DDPG = Memory(MEMORY_CAPACITY, dims=3 * STATE_DIM + ACTION_DIM + 2)

    saver = tf.train.Saver()
    path = './' + MODE[n_model]
    if LOAD:
        saver.restore(sess, tf.train.latest_checkpoint(path))
    else:
        sess.run(tf.global_variables_initializer())
    N_vals = np.linspace(0, 600, 600, endpoint=False)
    # N_vals = [1, 25, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600]
    N_trials = 1
    J_r_DDPG = np.zeros((N_trials, len(N_vals)))
    J_Curious = np.zeros((N_trials, len(N_vals)))
    J_r = np.zeros((N_trials, len(N_vals)))

    N_repeat = 10
    if LOAD:
        eval()
    else:
        train_DDPG()
        train()

        tot_samples = np.array(N_vals)
        colors = ['#2D328F', '#F15C19', "#81b13c", "#ca49ac"]

        label_fontsize = 18
        tick_fontsize = 14
        linewidth = 3
        markersize = 10
        plt.plot(tot_samples, np.amin(J_Curious, axis=0), 'o-', color=colors[0], linewidth=linewidth,
                 markersize=markersize, label='Curious_r')

        plt.axis([0, 600, 0, 200])
        plt.xlabel('rollouts', fontsize=label_fontsize)
        plt.ylabel('cost', fontsize=label_fontsize)
        plt.legend(fontsize=18, bbox_to_anchor=(1.0, 0.54))
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.grid(True)
        fig = plt.gcf()
        fig.set_size_inches(9, 6)

        plt.show()

        plt.plot(tot_samples, np.amin(J_r, axis=0), '-', color=colors[0], linewidth=linewidth,
                 markersize=markersize, label='Curious_GAN')
        plt.plot(tot_samples, np.amin(J_r_DDPG, axis=0), '-', color=colors[1], linewidth=linewidth,
                 markersize=markersize, label='DDPG')

        plt.axis([0, 600, 0, 200])
        plt.xlabel('rollouts', fontsize=label_fontsize)
        plt.ylabel('cost', fontsize=label_fontsize)
        plt.legend(fontsize=18, bbox_to_anchor=(1.0, 0.54))
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.grid(True)
        fig = plt.gcf()
        fig.set_size_inches(9, 6)

        plt.show()

        plt.plot(tot_samples, np.amin(J_D_loss, axis=0), 'o-', color=colors[0], linewidth=linewidth,
                 markersize=markersize, label='Curious_r')

        plt.axis([0, 600, 0, 1])
        plt.xlabel('rollouts', fontsize=label_fontsize)
        plt.ylabel('cost', fontsize=label_fontsize)
        plt.legend(fontsize=18, bbox_to_anchor=(1.0, 0.54))
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.grid(True)
        fig = plt.gcf()
        fig.set_size_inches(9, 6)

        plt.show()

        plt.plot(tot_samples, np.amin(J_G_loss, axis=0), 'o-', color=colors[0], linewidth=linewidth,
                 markersize=markersize, label='Curious_r')

        plt.axis([0, 600, 0, 1])
        plt.xlabel('rollouts', fontsize=label_fontsize)
        plt.ylabel('cost', fontsize=label_fontsize)
        plt.legend(fontsize=18, bbox_to_anchor=(1.0, 0.54))
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.grid(True)
        fig = plt.gcf()
        fig.set_size_inches(9, 6)

        plt.show()