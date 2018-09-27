import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer_conv2d
from tensorflow.contrib.layers import flatten

def conv_op(input_op, filter_size, channel_out, step, name):
    channel_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        weights = tf.get_variable(shape=[filter_size, filter_size, channel_in, channel_out], dtype=tf.float32,
                                  initializer=xavier_initializer_conv2d(), name=scope + 'weights')
        biases = tf.Variable(tf.constant(value=0.0, shape=[channel_out], dtype=tf.float32),
                             trainable=True, name='biases')
        conv = tf.nn.conv2d(input_op, weights, strides=[1, step, step, 1], padding='SAME') + biases
        conv = tf.nn.relu(conv, name=scope)
        return conv

def maxPool_op(input_op, filter_size, step, name):
    return tf.nn.max_pool(input_op, ksize=[1, filter_size, filter_size, 1], strides=[1, step, step, 1],
                          padding='SAME', name=name)

def full_connection(input_op, channel_out, name):
    channel_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        weight = tf.get_variable(shape=[channel_in, channel_out], dtype=tf.float32,
                                 initializer=xavier_initializer_conv2d(), name=scope + 'weight')
        bias = tf.Variable(tf.constant(value=0.0, shape=[channel_out], dtype=tf.float32), name='bias')
        fc = tf.nn.relu_layer(input_op, weight, bias, name=scope)
        return fc

def VGGNet_16(images, keep_prob):
    with tf.name_scope('block_1'):
        conv1_1 = conv_op(images, filter_size=3, channel_out=64, step=1, name='conv1_1')
        conv1_2 = conv_op(conv1_1, filter_size=3, channel_out=64, step=1, name='conv1_2')
        print('block1', conv1_2.shape)
        pool1 = maxPool_op(conv1_2, filter_size=2, step=2, name='pooling_1')
        print('pooling1', pool1.shape)

    with tf.name_scope('block_2'):
        conv2_1 = conv_op(pool1, filter_size=3, channel_out=128, step=1, name='conv2_1')
        conv2_2 = conv_op(conv2_1, filter_size=3, channel_out=128, step=1, name='conv2_2')
        print('block2', conv2_2.shape)
        pool2 = maxPool_op(conv2_2, filter_size=2, step=2, name='pooling_2')
        print('pooling2', pool2.shape)

    with tf.name_scope('block_3'):
        conv3_1 = conv_op(pool2, filter_size=3, channel_out=256, step=1, name='conv3_1')
        conv3_2 = conv_op(conv3_1, filter_size=3, channel_out=256, step=1, name='conv3_2')
        conv3_3 = conv_op(conv3_2, filter_size=3, channel_out=256, step=1, name='conv3_3')
        print('block3', conv3_3.shape)
        pool3 = maxPool_op(conv3_3, filter_size=2, step=2, name='pooling_3')
        print('pooling3', pool3.shape)

    with tf.name_scope('block_4'):
        conv4_1 = conv_op(pool3, filter_size=3, channel_out=512, step=1, name='conv4_1')
        conv4_2 = conv_op(conv4_1, filter_size=3, channel_out=512, step=1, name='conv4_2')
        conv4_3 = conv_op(conv4_2, filter_size=3, channel_out=512, step=1, name='conv4_3')
        print('block4', conv4_3.shape)
        pool4 = maxPool_op(conv4_3, filter_size=2, step=2, name='pooling_4')
        print('pooling4', pool4.shape)

    with tf.name_scope('block_5'):
        conv5_1 = conv_op(pool4, filter_size=3, channel_out=512, step=1, name='conv5_1')
        conv5_2 = conv_op(conv5_1, filter_size=3, channel_out=512, step=1, name='conv5_2')
        conv5_3 = conv_op(conv5_2, filter_size=3, channel_out=512, step=1, name='conv5_2')
        print('block5', conv5_3.shape)
        pool5 = maxPool_op(conv5_3, filter_size=2, step=2, name='pooling_5')
        print('pooling5', pool5.shape)

    #
    fc1 = flatten(pool5)
    dim = fc1.shape[1].value

    with tf.name_scope('FC1_4096'):
        fc1 = full_connection(fc1, channel_out=4096, name='FC1_4096')
        fc1_drop = tf.nn.dropout(fc1, keep_prob=keep_prob, name='fc1_drop')

    with tf.name_scope('FC2_4096'):
        fc2 = full_connection(fc1_drop, channel_out=4096, name='FC1_4096')
        fc2_drop = tf.nn.dropout(fc2, keep_prob=keep_prob, name='fc2_drop')

    with tf.name_scope('FC_1000'):
        fc3 = full_connection(fc2_drop, channel_out=1000, name='FC_1000')
        logits = tf.nn.softmax(fc3)
        return logits

batch_size = 128
def main(argv=None):
    images = tf.zeros(shape=[batch_size, 224, 224, 3])
    keep_prob = 0.8
    VGGNet_16(images, keep_prob)


if __name__ == '__main__':
    tf.app.run()