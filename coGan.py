from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import gzip,cPickle,sys
from keras.utils import np_utils

def dataset_load(path):
    if path.endswith(".gz"):
        f=gzip.open(path,'rb')
    else:
        f=open(path,'rb')

    if sys.version_info<(3,):
        data=cPickle.load(f)
    else:
        data=cPickle.load(f,encoding="bytes")
    f.close()
    return data


def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

#mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

xx,yy=dataset_load('./citynames.pkl.gz')

print(xx[0].shape)
test=[]
for i in range(len(xx)):
    img=xx[i]
    img=img.reshape(13056)
    test.append(img)
xx=np.asarray(test)
xx=xx.reshape(xx.shape[0],13056)



yy = np_utils.to_categorical(yy,70)

mb_size = 64
Z_dim = 100
X_dim = 13056
y_dim = 70

print(X_dim)
print(y_dim)


h_dim = 64
h_dim_2 = 128
h_dim_3 = 256
h_dim_4 = 256



def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


""" Discriminator Net model """
X = tf.placeholder(tf.float32, shape=[None, 13056])
y = tf.placeholder(tf.float32, shape=[None, y_dim])

D_W1 = tf.Variable(xavier_init([X_dim + y_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W2 = tf.Variable(xavier_init([h_dim, h_dim_2]))
D_b2 = tf.Variable(tf.zeros(shape=[h_dim_2]))

D_W3 = tf.Variable(xavier_init([h_dim_2, h_dim_3]))
D_b3 = tf.Variable(tf.zeros(shape=[h_dim_3]))

D_W4 = tf.Variable(xavier_init([h_dim_3, h_dim_4]))
D_b4 = tf.Variable(tf.zeros(shape=[h_dim_3]))

D_W5 = tf.Variable(xavier_init([h_dim_4, 1]))
D_b5 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_W3, D_W4,D_W5, D_b1, D_b2, D_b3, D_b4,D_b5]


def discriminator(x, y):
    inputs = tf.concat(axis=1, values=[x, y])

    print(inputs.shape)
    print(D_W1.shape)

    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1) # Input Layer
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2) # Hidden Layer 1
    D_h3 = tf.nn.relu(tf.matmul(D_h2, D_W3) + D_b3) # Hidden Layer 2
    D_h4 = tf.nn.relu(tf.matmul(D_h3, D_W4) + D_b4) # Hidden Layer 3

    D_logit = tf.matmul(D_h4, D_W4) + D_b4 # Output Layer

    D_prob = tf.nn.sigmoid(D_logit) # Finding the probability

    return D_prob, D_logit


""" Generator Net model """
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

G_W1 = tf.Variable(xavier_init([Z_dim + y_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

G_W2 = tf.Variable(xavier_init([h_dim, h_dim_2]))
G_b2 = tf.Variable(tf.zeros(shape=[h_dim_2]))

G_W3 = tf.Variable(xavier_init([h_dim_2, h_dim_3]))
G_b3 = tf.Variable(tf.zeros(shape=[h_dim_3]))

G_W4 = tf.Variable(xavier_init([h_dim_3, h_dim_4]))
G_b4 = tf.Variable(tf.zeros(shape=[h_dim_3]))

G_W5 = tf.Variable(xavier_init([h_dim_4, X_dim]))
G_b5 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_G = [G_W1, G_W2, G_W3, G_W4, G_W5, G_b1, G_b2,G_b3,G_b4,G_b5]


def generator(z, y):
    inputs = tf.concat(axis=1, values=[z, y])

    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    G_h3 = tf.nn.relu(tf.matmul(G_h2, G_W3) + G_b3)
    G_h4 = tf.nn.relu(tf.matmul(G_h3, G_W4) + G_b4)



    G_log_prob = tf.matmul(G_h4, G_W5) + G_b5

    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(68,192), cmap='Greys_r')
        # plt.imshow(sample.reshape(48,192), cmap='Greys_r')

    return fig


G_sample = generator(Z, y)
D_real, D_logit_real = discriminator(X, y)
D_fake, D_logit_fake = discriminator(G_sample, y)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0
batch=0
for it in range(1000000):
    if it % 1000 == 0:
        n_sample = 16

        Z_sample = sample_Z(n_sample, Z_dim)
        y_sample = np.zeros(shape=[n_sample, y_dim])
        y_sample[:, 0] = 1

        samples = sess.run(G_sample, feed_dict={Z: Z_sample, y:y_sample})

        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    # X_mb=[]
    # y_mb=[]
    # X_mb, y_mb = mnist.train.next_batch(mb_size)
    X_mb, y_mb = next_batch(64, xx, yy)

    # break
    Z_sample = sample_Z(mb_size, Z_dim)
    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: Z_sample, y:y_mb})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: Z_sample, y:y_mb})

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
print()
