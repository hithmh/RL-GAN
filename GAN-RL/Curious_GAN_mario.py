import tensorflow as tf
import numpy as np
import shutil
import os
from Conv_model import *
import matplotlib.pyplot as plt
import gym
import ppaquette_gym_super_mario
from ppaquette_gym_super_mario import wrappers
import env_wrapper
import multiprocessing
from PIL import Image


def train():
    # gpu_options = tf.GPUOptions(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85))
    var = 2.
    pointer = 0
    # timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
    # if timestep_limit is None: timestep_limit = env.spec.timestep_limit
    total_steps = 0
    lr_pointer = 0
    for ep in range(MAX_EPISODES):

        ep_reward = 0
        ep_curious_reward = 0
        D_loss = 0
        G_loss = 0
        l_2_loss = 0

        if ASYN_TRAIN_GAN:
            if M.pointer > MEMORY_CAPACITY:
                for t in range(ITER_D_Training):
                    b_s, b_g_old, b_lstm_s, b_a, b_r, b_curious_r, b_s_, b_lstm_s_ = M.sample(BATCH_SIZE)

                    if total_steps == LR_DECAY_LIST[lr_pointer]:
                        lr_pointer = min(lr_pointer + 1, len(LR_D_list))
                    generator.learn(b_s, b_a, LR_G_list[lr_pointer])
                    b_g = generator.predict_batch(b_s, b_a)

                    discriminator.learn(b_g, b_s_, b_s, b_a, LR_D_list[lr_pointer])
                    A_pred.learn(b_s, b_s_, b_a, LR_A_pred_list[lr_pointer])
                    if t%10 ==0:
                        one_step_D_loss = discriminator.eval(b_g, b_s_, b_s, b_a)
                        one_step_G_loss = generator.eval(b_s, b_a)
                        print('Ep:', ep,
                              '|D Training Step:%i'% int(t),
                              '| D loss: %f' % float(one_step_D_loss),
                              '| G loss: %f' % float(one_step_G_loss),
                              )
        lstm_state = actor.LSTM_unit.get_initial_state()
        env = gym.make(env_id)
        env = env_wrapper.BufferedObsEnv(env, n=TIME_STEP, skip=frame_skip, shape=fshape, channel_last=True)
        s = env.reset()

        a_pred_loss= 0
        for t in range(MAX_EP_STEPS):
            a = actor.choose_action(s, lstm_state, var)
            g = generator.predict(s, a)
            s_, r, done, info, _ = env._step(a)

            curious_r = ITA * discriminator.determine(s, a, g)[0]
            lstm_state_ = actor.LSTM_unit.get_state(s, lstm_state)

            one_step_l_2_loss = discriminator.observe_and_compare(s_, g,)
            l_2_loss += one_step_l_2_loss

            M.store_transition(s, g, lstm_state, a, r, curious_r, s_,lstm_state_)
            if M.pointer > MEMORY_CAPACITY:
                var = max([var * .9999, VAR_MIN])
                # for i in range(ITER_train_G):
                #     b_s, b_a, b_r, b_curious_r, b_s_ = M.sample(BATCH_SIZE)
                #     generator.learn(b_s, b_a)

                if total_steps == LR_DECAY_LIST[min(lr_pointer,len(LR_DECAY_LIST)-1)]:
                    lr_pointer = min(lr_pointer+1,len(LR_D_list))


                b_s, b_g_old, b_lstm_s, b_a, b_r, b_curious_r, b_s_, b_lstm_s_ = M.sample(BATCH_SIZE)
                if not ASYN_TRAIN_GAN:
                    generator.learn(b_s, b_a, LR_G_list[lr_pointer], b_s_)
                b_g = generator.predict_batch(b_s, b_a)
                if not ASYN_TRAIN_GAN:
                    discriminator.learn(b_g, b_s_, b_s, b_a, LR_D_list[lr_pointer])

                # Learn the minibatch
                # b_curious_r = discriminator.determine_batch(b_s, b_lstm_s, b_a, b_g)
                A_pred.learn(b_s, b_s_, b_a, LR_A_pred_list[lr_pointer])
                critic.learn(b_s,b_lstm_s, b_a, b_curious_r, b_s_, b_lstm_s_)
                actor.learn(b_s, b_lstm_s)
                one_step_D_loss = discriminator.eval(b_g, b_s_, b_s, b_a)
                D_loss += one_step_D_loss
                one_step_G_loss = generator.eval(b_s, b_a, b_s_)
                G_loss += one_step_G_loss
                a_pred_loss = A_pred.eval(b_s, b_s_, b_a)


                if t % 10 == 0:
                    check_real_data = discriminator.check_the_real_data(s, a, s_)
                    print('Ep:', ep,
                          '| Step:%i' % int(t),
                          '| R: %i' % int(ep_reward),
                          '| Curious_R: %f' % float(curious_r),
                          '| Real_data_Check: %f' % float(check_real_data),
                          '| Prediction_error: %f' % float(one_step_l_2_loss),
                          '| A_Pred_error: %f' % float(a_pred_loss),
                          '| D loss: %f' % float(one_step_D_loss),
                          '| G loss: %f' % float(one_step_G_loss),
                          '| Explore: %.2f' % var,
                          '| Global Steps:%i' % int(total_steps),
                              )

                if n_mode==1 and t % 200 == 0:
                        plt.ion()
                        plt.imshow(transfer_picture([s_[-1], g[0]]), cmap='gray')
                        plt.pause(1)
                        plt.close()

            s = s_
            lstm_state = lstm_state_
            ep_reward += r
            ep_curious_reward += curious_r
            total_steps += 1
    
            if t == MAX_EP_STEPS - 1 or done or info['life']==0 or info['time']<=1:
                # if done:
                t = t+1
                result = '| done' if done else '| ----'
                print('Ep:', ep,
                      result,
                      '| R: %i' % int(ep_reward),
                      '| Curious_R: %f' % float(ep_curious_reward),
                      '| D_loss: %f' % float(D_loss/t),
                      '| G_loss: %f' % float(G_loss/t),
                      '| Prediction_error: %f' % float(l_2_loss/t),
                      '| Explore: %.2f' % var,
                      )
                # plt.ion()

                env.close()

                break

        if ep == N_vals[pointer]:

            # evaluate the minibatch
            J_A_pred_loss[0, pointer] = a_pred_loss
            J_D_loss[0, pointer] = D_loss/t
            J_G_loss[0, pointer] = G_loss/t
            J_Curious[0,pointer] = ep_curious_reward
            J_r[0, pointer] = ep_reward
            if pointer<N_vals.__len__()-1:
                pointer += 1

                
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.mkdir(path)
    ckpt_path = os.path.join('./'+MODE[n_mode], 'Curious_GAN.ckpt')
    save_path = saver.save(sess, ckpt_path, write_meta_graph=False)
    print("\nSave Model %s\n" % save_path)
    np.save('J_r.npy', J_r)


def GAN_pretrain():
    #collect data
    var=2
    env = gym.make(env_id)
    env = env_wrapper.BufferedObsEnv(env, n=TIME_STEP, skip=frame_skip, shape=fshape, channel_last=False)
    s = env.reset()
    s = np.expand_dims(s, -1)
    lstm_state = actor.LSTM_unit.get_initial_state()
    lr_pointer = 0
    for t in range(MAX_EP_STEPS):
        a = actor.choose_action(s, lstm_state, var)
        g = generator.predict(s, a)
        s_, r, done, info, _ = env._step(a)

        curious_r = ITA * discriminator.determine(s, a, g)[0]
        lstm_state_ = actor.LSTM_unit.get_state(s, lstm_state)

        one_step_l_2_loss = discriminator.observe_and_compare(s_, g, )
        l_2_loss += one_step_l_2_loss
        M.store_transition(s, g, lstm_state, a, r, curious_r, s_, lstm_state_)
        s = s_
        lstm_state = lstm_state_

    for t in range(MAX_PRETRAIN_STEPS):
        b_s, b_g_old, b_lstm_s, b_a, b_r, b_curious_r, b_s_, b_lstm_s_ = M.sample(BATCH_SIZE)
        generator.learn(b_s, b_a, LR_G_list[lr_pointer])
        b_g = generator.predict_batch(b_s, b_a)

        discriminator.learn(b_g, b_s_, b_s, b_a, LR_D_list[lr_pointer])
        A_pred.learn(b_s, b_s_, b_a, LR_A_pred_list[lr_pointer])
        if t % 10 == 0:
            one_step_D_loss = discriminator.eval(b_g, b_s_, b_s, b_a)
            one_step_G_loss = generator.eval(b_s, b_a)
            print('Ep:', ep,
                  '| Pre-Training Step:%i' % int(t),
                  '| D loss: %f' % float(one_step_D_loss),
                  '| G loss: %f' % float(one_step_G_loss),
                  )


        if n_mode == 1 and t % 200 == 0:
            plt.close()
            plt.ion()
            plt.imshow(transfer_picture([b_s_[0,-1], b_g[0,0]]), cmap='gray')
            # plt.pause(1)

            plt.show()




def transfer_picture(images, mode='L'):
    w = images[0].shape[0]
    h = images[0].shape[1]
    new_im = Image.new(mode, (w * 2, h))
    col=0
    for image in images:

        image1 = image[ :, :, 0]
        image1 = (((image1 - image1.min()) * 255) / (image1.max() - image1.min())).astype(np.uint8)
        new_im.paste(Image.fromarray(image1,mode),(col,0))
        col += w
    return new_im



def eval():
    env.set_fps(30)
    s = env.reset()
    while True:
        if RENDER:
            env.render()
        a = actor.choose_action(s)
        s_, r, done = env.step(a)
        s = s_


def build_model():

    with tf.name_scope("S"):
        S = tf.placeholder(tf.float32, shape=[None, *list(env.observation_space.shape)], name="s")
    # with tf.name_scope("fake_S_"):
    #     fake_S_ = tf.placeholder(tf.float32, shape=[None, 1, *fshape, 1], name="fake_s_")
    with tf.name_scope('R'):
        R = tf.placeholder(tf.float32, [None, 1], name='r')
    with tf.name_scope('S_'):
        S_ = tf.placeholder(tf.float32, shape=[None, *list(env.observation_space.shape)], name='s_')
    with tf.name_scope('A'):
        A = tf.placeholder(tf.float32, shape=[None, ACTION_DIM], name='a')
    with tf.name_scope('LR'):
        LR_A_pred = tf.placeholder(tf.float32)
        LR_D = tf.placeholder(tf.float32)
        LR_G = tf.placeholder(tf.float32)
    with tf.name_scope('Is_training'):
        phi_is_training = tf.placeholder(tf.bool)

        D_is_training = tf.placeholder(tf.bool)
        G_is_training = tf.placeholder(tf.bool)
        C_is_training = tf.placeholder(tf.bool)
        A_is_training = tf.placeholder(tf.bool)
        A_pred_is_training = tf.placeholder(tf.bool)

    with tf.variable_scope('feature_abstraction'):
        phis = universe_feature_abstraction(S, time_step=TIME_STEP, nConvs=2,is_training=phi_is_training)
        phis_ = universe_feature_abstraction(S_, reuse=True, time_step=TIME_STEP, nConvs=2, is_training=False)
        # observation = universe_feature_abstraction(single_S_, reuse=True, time_step=1, nConvs=2, is_training=False)
        # observation = tf.squeeze(observation, 1)


    A_pred = StateActionPredictor(S, S_, phis, phis_, A, A_pred_is_training,phi_is_training, l2_weight, LR_A_pred, sess)
    actor = Actor(sess, ACTION_DIM, LR_A, BATCH_SIZE, REPLACE_ITER_A, phis, phis_, S, S_,
                  phi_is_training, A_is_training, C_is_training, USE_BATCH_NORM, scope='Actor', action_bound=None)

    critic = Critic(sess, ACTION_DIM, LR_C, BATCH_SIZE, GAMMA, REPLACE_ITER_C, actor, phis, phis_, S, S_, R,
                    phi_is_training, C_is_training, A_is_training, USE_BATCH_NORM)

    actor.add_grad_to_graph(critic.a_grads)
    M = Memory(capacity=MEMORY_CAPACITY)
    generator = Generator(sess, ACTION_DIM, LR_G, BATCH_SIZE, actor.a, S,S_, phis, A,
                          l2_weight, phi_is_training, G_is_training, A_is_training, D_is_training, MODE[n_mode])
    if n_mode == 1:

        with tf.variable_scope('D_feature_abstraction'):
            fake_observation = universe_feature_abstraction(generator.G, reuse=True, time_step=1, nConvs=2, is_training=False)
            fake_observation = tf.squeeze(fake_observation, 1)

        discriminator = Discriminator(sess, LR_D, S_, S, fake_observation, generator.G, actor.a, LSTM_unit, gen_LSTM_unit, observation,
                                  phi_is_training, G_is_training, A_is_training, D_is_training, BATCH_SIZE, l2_weight,
                                  MODE[n_mode])
    else:
        discriminator = Discriminator(sess, LR_D, S_, S, phis, generator.G, generator.G, actor.a,
                                      phis_,
                                      phi_is_training, G_is_training, A_is_training, D_is_training, BATCH_SIZE,
                                      l2_weight,
                                      MODE[n_mode])
    generator.model_loss(discriminator.unscaled_D_fake, phis_)
    return actor, critic, generator, discriminator, M, A_pred




if __name__ == '__main__':
    env_id = 'ppaquette/SuperMarioBros-1-1-v0'

    smooth = 0
    np.random.seed(1)
    # tf.set_random_seed(1)
    acRepeat = 0
    fshape = (42, 42)
    TIME_STEP = 4
    MAX_EPISODES = 100
    MAX_EP_STEPS = 4000
    MAX_PRETRAIN_STEPS = 10000
    USE_PRETRAIN = False
    LR_DECAY_LIST = [14000,50000]
    LR_D_list = [1e-4, 1e-6, 1e-8]  # learning rate for actor
    LR_G_list = [1e-4, 1e-6, 1e-8]
    LR_A_pred_list = [1e-4, 1e-6, 1e-8]# learning rate for critic
    LR_A = 1e-4  # learning rate for actor
    LR_C = 1e-4  # learning rate for critic
    GAMMA = 0.99  # reward discount
    l2_weight = 0.01
    USE_BATCH_NORM = True
    ASYN_TRAIN_GAN = False
    REPLACE_ITER_A = 1100
    REPLACE_ITER_C = 1000
    ITER_train_G = 10
    ITER_D_Training = 2000
    ITER_G_Training =400
    MEMORY_CAPACITY = 5000
    BATCH_SIZE = 32
    VAR_MIN = 1
    RENDER = False
    LOAD = False
    MODE = ['hidden_state', 'full_prediction']
    n_mode = 0
    ITA = 1  # Curious coefficient

    frame_skip = acRepeat if acRepeat > 0 else 4
    lock = multiprocessing.Lock()
    env = gym.make(env_id)
    env.configure(lock=lock)
    env = env_wrapper.BufferedObsEnv(env, n=TIME_STEP, skip=frame_skip, shape=fshape, channel_last=True)

    STATE_DIM = env.observation_space.shape
    ACTION_DIM = env.action_space.shape

    sess = tf.Session()

    actor, critic, generator, discriminator, M, A_pred= build_model()

    saver = tf.train.Saver()
    path = './' + MODE[n_mode]
    if LOAD:
        saver.restore(sess, tf.train.latest_checkpoint(path))
    else:
        sess.run(tf.global_variables_initializer())

    N_vals = np.linspace(0, MAX_EPISODES, MAX_EPISODES, endpoint=False)
    # N_vals = [1, 25, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600]
    N_trials = 1
    J_r_DDPG = np.zeros((N_trials, len(N_vals)))
    J_Curious = np.zeros((N_trials, len(N_vals)))
    J_r = np.zeros((N_trials, len(N_vals)))
    J_A_pred_loss = np.zeros((N_trials, len(N_vals)))
    J_D_loss = np.zeros((N_trials, len(N_vals)))
    J_G_loss = np.zeros((N_trials, len(N_vals)))
    env.close
    N_repeat = 10
    # if LOAD:
    #     eval()
    # else:
    if True:
        if USE_PRETRAIN:
            GAN_pretrain()
        train()

        tot_samples = np.array(N_vals)
        colors = ['#2D328F', '#F15C19', "#81b13c", "#ca49ac"]

        label_fontsize = 18
        tick_fontsize = 14
        linewidth = 1
        markersize = 10

        plt.plot(tot_samples, np.amin(J_r, axis=0), 'o-', color=colors[0], linewidth=linewidth,
                 markersize=markersize, label='R')

        plt.axis([0, MAX_EPISODES, 0, 800])
        plt.xlabel('rollouts', fontsize=label_fontsize)
        plt.ylabel('cost', fontsize=label_fontsize)
        plt.legend(fontsize=18, bbox_to_anchor=(1.0, 0.54))
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.grid(True)
        fig = plt.gcf()
        fig.set_size_inches(9, 6)

        plt.show()

        plt.plot(tot_samples, np.amin(J_Curious, axis=0), 'o-', color=colors[0], linewidth=linewidth,
                 markersize=markersize, label='Curious_r')

        plt.axis([0, MAX_EPISODES, -0.5, 0.5])
        plt.xlabel('rollouts', fontsize=label_fontsize)
        plt.ylabel('cost', fontsize=label_fontsize)
        plt.legend(fontsize=18, bbox_to_anchor=(1.0, 0.54))
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.grid(True)
        fig = plt.gcf()
        fig.set_size_inches(9, 6)

        plt.show()

        plt.plot(tot_samples, np.amin(J_A_pred_loss, axis=0), 'o-', color=colors[0], linewidth=linewidth,
                 markersize=markersize, label='A_pred_loss')

        plt.axis([0, MAX_EPISODES, 0, 5])
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
        # plt.plot(tot_samples, np.amin(J_r_DDPG, axis=0), '-', color=colors[1], linewidth=linewidth,
        #          markersize=markersize, label='DDPG')

        plt.axis([0, MAX_EPISODES, 0, 900])
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

        plt.axis([0, MAX_EPISODES, 0, 1])
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

        plt.axis([0, MAX_EPISODES, 0, 1])
        plt.xlabel('rollouts', fontsize=label_fontsize)
        plt.ylabel('cost', fontsize=label_fontsize)
        plt.legend(fontsize=18, bbox_to_anchor=(1.0, 0.54))
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.grid(True)
        fig = plt.gcf()
        fig.set_size_inches(9, 6)

        plt.show()