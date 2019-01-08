import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import numpy as np
from collections import deque
import random

BELTA = 0.0003
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),  # KL penalty
    # Clipped surrogate objective, find this is better
    dict(name='clip', epsilon=0.1),
][1]  # choose the method for optimization


class PPO(object):
    replay_memory = deque()
    memory_size = 100

    def __init__(self, S_DIM, A_DIM, BATCH, A_UPDATE_STEPS, C_UPDATE_STEPS, HAVE_TRAIN, num):  # num是什么意思
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')
        self.S_DIM = S_DIM
        self.A_DIM = A_DIM
        self.BATCH = BATCH
        self.A_UPDATE_STEPS = A_UPDATE_STEPS
        self.C_UPDATE_STEPS = C_UPDATE_STEPS
        self.decay = tf.placeholder(tf.float32, (), 'decay')
        self.a_lr = tf.placeholder(tf.float32, (), 'a_lr')
        self.c_lr = tf.placeholder(tf.float32, (), 'c_lr')
        self.num = num

        # critic
        with tf.variable_scope('critic'):
            w1 = tf.Variable(tf.truncated_normal(
                [self.S_DIM, 200], stddev=0.01), name='w1')
            bias1 = tf.Variable(tf.constant(
                0.0, shape=[200], dtype=tf.float32), name='b1')
            l1 = tf.nn.relu(tf.matmul(self.tfs, w1) + bias1)
            # l1 = tf.reshape(l1, shape=(-1, 200, 1))
            # lstm_cell = rnn.BasicLSTMCell(num_units=128)
            # # init_state = lstm_cell.zero_state(
            # # batch_size=self.BATCH, dtype=tf.float32)
            # outputs, states = tf.nn.dynamic_rnn(
            #     cell=lstm_cell, inputs=l1, dtype=tf.float32)  # , initial_state=init_state, dtype=tf.float32)
            # l2 = outputs[:, -1, :]
            # print(np.shape(l2))

            w2 = tf.Variable(tf.truncated_normal(
                [200, 50], stddev=0.01), name='w2')
            bias2 = tf.Variable(tf.constant(
                0.0, shape=[50], dtype=tf.float32), name='b2')
            l2 = tf.nn.relu(tf.matmul(l1, w2) + bias2)

            w3 = tf.Variable(tf.truncated_normal(
                [50, 1], stddev=0.01), name='w3')
            bias3 = tf.Variable(tf.constant(
                0.0, shape=[1], dtype=tf.float32), name='b3')
            self.v = tf.nn.relu(tf.matmul(l2, w3) + bias3)

            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage)) + \
                BELTA * (tf.nn.l2_loss(w1) + tf.nn.l2_loss(w3))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.c_lr)
            vars_ = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(
                tf.gradients(self.closs, vars_), 5.0)
            self.ctrain_op = optimizer.apply_gradients(zip(grads, vars_))

        # actor
        pi, pi_params, l2_loss_a = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params, _ = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            # choosing action  squeeze:去掉第一维（=1）
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(
                p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, self.A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                ratio = pi.prob(self.tfa) #/oldpi.prob(self.tfa)
                surr = ratio * self.tfadv
            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:  # clipping method, find this is better
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1. - METHOD['epsilon'], 1. + METHOD['epsilon']) * self.tfadv)) + \
                    BELTA * l2_loss_a

        with tf.variable_scope('atrain'):
            # self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.a_lr)
            vars_ = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(
                tf.gradients(self.aloss, vars_), 5.0)
            self.atrain_op = optimizer.apply_gradients(zip(grads, vars_))

        tf.summary.FileWriter("log/", self.sess.graph)
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('tmp/vintf/', self.sess.graph)
        self.sess.run(init)
        if HAVE_TRAIN == True:
            model_file = tf.train.latest_checkpoint(
                'ckpt/' + str(self.num) + "/")
            self.saver.restore(self.sess, model_file)

    def update(self, s, a, r, dec, alr, clr, epoch):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(
            self.advantage, {self.tfs: s, self.tfdc_r: r, self.decay: dec})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(self.A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam']})
                if kl > 4 * METHOD['kl_target']:  # this in in google's paper
                    break
            # adaptive lambda, this is in OpenAI's paper
            if kl < METHOD['kl_target'] / 1.5:
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            # sometimes explode, this clipping is my solution
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)
        else:  # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op,
                           {self.tfs: s, self.tfa: a, self.tfadv: adv, self.decay: dec, self.a_lr: alr, self.c_lr: clr})
             for _ in range(self.A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r, self.decay: dec, self.a_lr: alr, self.c_lr: clr})
         for _ in range(self.C_UPDATE_STEPS)]
        # self.saver.save(self.sess, "ckpt/" + str(self.num) + "/", global_step=epoch)

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            w4 = tf.Variable(tf.truncated_normal(
                [self.S_DIM, 200], stddev=0.01), name='w4')
            bias4 = tf.Variable(tf.constant(
                0.0, shape=[200], dtype=tf.float32), name='b4')
            l3 = tf.nn.sigmoid(tf.matmul(self.tfs, w4) + bias4)

            w5 = tf.Variable(tf.truncated_normal(
                [200, 50], stddev=0.01), name='w5')
            bias5 = tf.Variable(tf.constant(
                0.0, shape=[50], dtype=tf.float32), name='b5')
            l4 = tf.nn.sigmoid(tf.matmul(l3, w5) + bias5)

            w6 = tf.Variable(tf.truncated_normal(
                [50, self.A_DIM], stddev=0.01), name='w6')
            bias6 = tf.Variable(tf.constant(
                0.0, shape=[self.A_DIM], dtype=tf.float32), name='b6')

            mu = 1 * tf.nn.sigmoid(tf.matmul(l4, w6) + bias6)
            # mu = 5 * tf.nn.sigmoid(tf.matmul(l4, w6) + bias6) + 0.0001
            # print('mu:', np.shape(mu))

            w7 = tf.Variable(tf.truncated_normal(
                [50, self.A_DIM], stddev=0.01), name='w7')
            bias7 = tf.Variable(tf.constant(
                0.0, shape=[self.A_DIM], dtype=tf.float32), name='b7')
            sigma = self.decay * \
                tf.nn.sigmoid(tf.matmul(l4, w7) + bias7) + 0.00001
            # print('sigma:',np.shape(sigma))

            # mu = tf.layers.dense(l2, A_DIM, tf.nn.sigmoid, trainable=trainable)
            # sigma = tf.layers.dense(l2, A_DIM, tf.nn.sigmoid, trainable=trainable) + 0.0001
            norm_dist = tf.distributions.Normal(
                loc=mu, scale=sigma)  # loc：mean  scale：sigma
        params = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=name)  # 返回name=="name"的变量列表
        l2_loss_a = tf.nn.l2_loss(
            w4) + tf.nn.l2_loss(w5) + tf.nn.l2_loss(w6) + tf.nn.l2_loss(w7)
        return norm_dist, params, l2_loss_a

    def choose_action(self, s, dec):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, feed_dict={
                          self.tfs: s, self.decay: dec})
        # a, sigma, mu = self.sess.run([self.sample_op, self.sigma, self.mu], feed_dict={self.tfs: s, self.decay: dec})

        return np.clip(a[0], 0.0001, 1)  # 把输出限制在[0, 5]之内

    def get_v(self, s):
        if s.ndim < 2:
            s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

    def preceive(self, state, action, reward, dec, alr, clr):
        self.replay_memory.append((state, action, reward))
        if len(self.replay_memory) > self.memory_size:
            self.replay_memory.popleft()
        else:
            self.train_network(dec, alr, clr)

    def train_network(self, dec, alr, clr):
        mini_batch = random.sample(self.replay_memory, self.BATCH)
        state_batch = [data[0] for data in mini_batch]
        action_batch = [data[1] for data in mini_batch]
        reward_batch = [data[2] for data in mini_batch]

        self.update(state_batch, action_batch, reward_batch, dec, alr, clr)
