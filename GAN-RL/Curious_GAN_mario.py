import tensorflow as tf
import numpy as np
import shutil
import os
from Conv_model import Discriminator, Generator, Memory, Actor, Critic
import matplotlib.pyplot as plt
import gym
import ppaquette_gym_super_mario
from ppaquette_gym_super_mario import wrappers
import env_wrapper


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

            a = actor.choose_action(s,var)

            g = generator.predict(s, a)
            s_, r, done, info, _ = env._step(a)
            curious_r = r + ITA * discriminator.determine(s,a,g)[0]

            l_2_loss += np.sum(np.square(actor.observe(s_)-g))

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
    env_id = 'ppaquette/SuperMarioBros-1-1-v0'

    smooth = 0
    np.random.seed(1)
    # tf.set_random_seed(1)
    acRepeat = 0
    fshape = (42, 42)
    TIME_STEP = 4
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

    frame_skip = acRepeat if acRepeat > 0 else 4

    env = gym.make(env_id)
    env = env_wrapper.BufferedObsEnv(env, n=1, skip=frame_skip, shape=fshape)

    STATE_DIM = env.observation_space.shape
    ACTION_DIM = env.action_space.shape
    S = []
    with tf.name_scope("S"):
        S=tf.placeholder(tf.float32, shape=[None, *STATE_DIM], name="s")
    for i in range(TIME_STEP):
        with tf.name_scope("S_t-{}".format(i)):
            S.append(tf.placeholder(tf.float32, shape=[None, *STATE_DIM], name="s_t-{}".format(i)))
    with tf.name_scope('R'):
        R = tf.placeholder(tf.float32, [None, 1], name='r')
    with tf.name_scope('S_'):
        S_ = tf.placeholder(tf.float32, shape=[None,TIME_STEP, *STATE_DIM], name='s_')


    sess = tf.Session()
    actor = Actor(sess, ACTION_DIM, learning_rate=LR_A, t_replace_iter=REPLACE_ITER_A, S=S, S_=S_, time_step = TIME_STEP)
    critic = Critic(sess, ACTION_DIM, LR_C, GAMMA, REPLACE_ITER_C, actor.a, actor.a_, S=S, S_=S_, R=R)
    actor.add_grad_to_graph(critic.a_grads)
    M = Memory(capacity = MEMORY_CAPACITY)
    generator = Generator(Learning_rate=LR_G, a=actor.a, S=S, action_dim=ACTION_DIM, sess=sess)
    discriminator = Discriminator(G=generator.G, Learning_rate=LR_D, S=S, S_=S_, a=actor.a,
                                  sess=sess)
    generator.model_loss(discriminator.D_logit_fake)

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